# Author: Chenhongyi Yang

from abc import ABCMeta
from typing import Tuple

import copy
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import normal_

from mmdet.models.detectors.deformable_detr import MultiScaleDeformableAttention

from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig

from mmdet.models.detectors.base import BaseDetector

from mmengine.model import xavier_init

from mmdet.structures.det_data_sample import DetDataSample
from mmengine.structures.instance_data import InstanceData

from mmdet.models.layers import (DeformableDetrTransformerDecoder,
                      DeformableDetrTransformerEncoder, SinePositionalEncoding)

from projects.RefSAM.models.segment_anything import build as sam_builder
import numpy as np

class RefTokenFuserMLP(nn.Module):
    def __init__(self, embed_dims, num_tokens):
        super(RefTokenFuserMLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embed_dims * num_tokens, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, embed_dims)
        )

    def forward(self, tokens):
        b = tokens.shape[0]
        x = tokens.reshape(b, -1)
        x = self.mlp(x)
        return x

@MODELS.register_module()
class RefSAMDetector(BaseDetector, metaclass=ABCMeta):

    def __init__(
        self,
        sam: ConfigType,
        embed_dims=None,

        mode: str = None,  # None (training), 'compute_embeds', 'eval_embeds' (eval with precomputed npy)
        embeds_path: str = None,  # Path to save or load the embeddings
        n_shot: int = None,  # n-shot when mode='compute_embeds' is the number of reference images per class
                             # when mode='eval_embeds' is the number of reference images to use for evaluation

        neck: OptConfigType = None,
        encoder: OptConfigType = None,
        decoder: OptConfigType = None,
        bbox_head: OptConfigType = None,
        positional_encoding: OptConfigType = None,
        num_queries: int = 100,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,

        with_box_refine: bool = False,
        num_feature_levels: int = 4,
    ) -> None:

        super().__init__(
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor
        )

        self.with_box_refine = with_box_refine
        self.num_feature_levels = num_feature_levels
        self.sam_type = sam.get("type", None)
        self.embed_dims = embed_dims
        assert self.sam_type is not None

        self.sam = sam_builder.sam_registry[self.sam_type](sam.get("checkpoint", None))

        if bbox_head is not None:
            assert 'share_pred_layer' not in bbox_head
            assert 'num_pred_layer' not in bbox_head
            assert 'as_two_stage' not in bbox_head

            bbox_head['share_pred_layer'] = not with_box_refine
            bbox_head['num_pred_layer'] = decoder['num_layers']
            bbox_head['as_two_stage'] = False

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.encoder = encoder
        self.decoder = decoder
        self.positional_encoding = positional_encoding
        self.num_queries = num_queries

        # Custom defined parameters
        self.mode = mode
        self.embeds_path = embeds_path
        self.n_shot = n_shot

        # Cache for storing precomputed queries and labels for the reference images
        self.num_categories = 80
        self.ref_index = 0  # Index to keep track of position in the cache
        self.num_masks_outs = 5 if 'hq' in self.sam_type else 4
        if self.mode == 'compute_embeds' or self.mode == 'eval_embeds':
            # Create cache for queries and labels
            self.register_buffer('query_cache',
                    torch.zeros((self.num_categories*self.n_shot, self.num_masks_outs, embed_dims)))
            self.register_buffer('label_cache', torch.zeros((self.num_categories*self.n_shot, 1)))
            if self.mode == 'eval_embeds':
                # Load precomputed embeddings
                data = np.load(self.embeds_path, allow_pickle=True).item()  # .item() to load dict
                cached_queries = torch.from_numpy(data['query_cache'])
                cached_labels = torch.from_numpy(data['label_cache'])
                self.query_cache[:] = cached_queries
                self.label_cache[:] = cached_labels
                print("Loaded precomputed queries. Shape:", self.query_cache.shape)
                print("Loaded precomputed labels. Shape:", self.label_cache.shape)
                assert self.n_shot * self.num_categories <= self.query_cache.shape[0], \
                    f"There are {self.num_categories} categories, and a total of {self.query_cache.shape[0]} \
                        items. Thus, max n-shot setting is {self.query_cache.shape[0] // self.num_categories}"

        # init model layers
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.bbox_head = MODELS.build(bbox_head)
        self._init_layers()

        num_mask_tokens = 5 if 'hq' in self.sam_type else 4
        self.ref_tokens_fuser = RefTokenFuserMLP(self.embed_dims, num_mask_tokens)

        for param in self.sam.parameters():
            param.requires_grad = False

    def _init_layers(self):
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)
        if self.encoder is not None:
            self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = DeformableDetrTransformerDecoder(**self.decoder)

        if self.embed_dims is None:
            self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries,
                                            self.embed_dims * 2)
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.reference_points_fc = nn.Linear(self.embed_dims, 2)

    def init_weights(self):
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        xavier_init(
            self.reference_points_fc, distribution='uniform', bias=0.)
        normal_(self.level_embed)

    def forward(
        self,
        inputs: torch.Tensor,
        data_samples: OptSampleList = None,
        mode: str = 'tensor'
    ):

        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise (f'Invalid mode "{mode}". Only supports loss, predict and tensor mode')

    def _forward(
        self,
        batch_inputs: Tensor,
        batch_data_samples: OptSampleList # SampleList
    ):
        """
        Args:
            batch_inputs (Tensor): [B, C, H, W] Target images
            batch_data_samples (List[DetDataSample]): List of reference images [B,]
                NOTE: Reference images sampled will be different every iteration
                      (For example, target_image 1 may have 3 reference images in one iteration)

        Shape and dict key information:
            batch_inputs: [B, C, H, W]
            batch_data_samples: List of DetDataSample [B,]
                DetDataSample:
                    gt_instances: InstanceData
                        labels: [n_gt]
                        bboxes: [n_gt, 4]
                        masks: [n_gt, H, W]
                    ref_data: InstanceData
                        ref_images: [n_ref, 3, img_h, img_w]
                        ref_data_samples: [n_ref, InstanceData]
                            InstanceData:
                                labels: [n_gt]
                                bboxes: [n_gt, 4]
                                masks: [n_gt, H, W] (datatype: BitmapMasks)
        """
        # PIL images are already loaded into GPU by now :(
        # print("batch_inputs.shape:", batch_inputs.shape)
        # print("batch_inputs[0][0]", batch_inputs[0][0])
        # print("batch_inputs[0]", batch_inputs[0])
        # exit()

        device = batch_inputs.device
        batch_size = batch_inputs.shape[0] # (batch_size) x 3 x 1024 x 1024      
        
        # === First Batch Visualisation ===
        # self.plot_batch_img4embeddings(batch_inputs, batch_data_samples)
        # self.plot_batch(batch_inputs, batch_data_samples)
        # exit()
        
        # Check that the labels are the same for the reference images and the target image
        # assert batch_data_samples[0].gt_instances.labels == batch_data_samples[0].ref_data.ref_data_samples[0].labels
        if self.mode == 'eval_embeds':
            # We don't have ref images because we load them from precomputed npy file
            with torch.no_grad():
                all_sam_features, all_interm_features = self._extract_sam_features(batch_inputs)
        
        else:
            # If here then: (1) We are training or (2) Computing npy embeddings for ref images
            
            # === Extract data from batch_data_samples ===

            # ref_images stores the reference images for each sample in the batch
            # shape of ref_images: [batch_size, n_ref, 3, img_h, img_w]
            ref_images = [x.ref_data.ref_images.to(dtype=batch_inputs.dtype) for x in batch_data_samples]   # each [n_ref, 3, img_h, img_w]
            # print("batch_inputs.shape:", batch_inputs.shape)
            # print([x.shape for x in ref_images])
            # ref_samples stores the GT instances for each reference image in the batch
            # shape of ref_samples: [batch_size, n_ref, InstanceData]
            ref_samples = [x.ref_data.ref_data_samples for x in batch_data_samples]  # each [n_ref of GT instances]
            
            # === Housekeeping ===

            # Sample prompts for each reference image (bounding box only for now)
            ref_prompts = [self._sample_prompts(x, only_box=True) for x in ref_samples]

            # === SAM image encoder ===
            
            # Forward target images and its corresponding reference images
            # The first B features correspond to the batch_inputs=target_images, 
            # The other features correspond to the reference images
            with torch.no_grad():
                # all_sam_features: correspond to the final features of the SAM image encoder
                # all_interm_features: correspond to the intermediate features at each layer of the SAM image encoder
                all_sam_features, all_interm_features = self._extract_sam_features(
                    torch.cat([batch_inputs] + ref_images, dim=0) # ref_images is a list so we do list concatenation
                )
                # all_sam_features: [B+n_ref, out_channels, H_dim, W_dim]
                # for example, if batch_size=3 and n_ref=[1, 4, 1], then
                #                   all_sam_features.shape = [3+6, out_channels=256, H_dim=64, W_dim=64]
                
                # all_interm_features: [num_feature_levels=4, B+n_ref=9, H_dim=64, W_dim=64, hidden_dim_channels=768]

            # FINAL FEATURES: corresponding only to the reference images
            ref_sam_featuers = all_sam_features[batch_size:]
            # INTERMEDIATE FEATURES: corresponding to the reference images
            ref_interm_features = [x[batch_size:] for x in all_interm_features]

            '''
                Batch list, each [n_ref, num_mask_outs, 256]
                
                len(num_masks_outs) = 5
                - 0: single_mask_token
                - 1, 2, 3: multi_mask_token
                - 4: hq_mask_token (optional)
            '''
            
            # === Forward the reference images into SAM decoder ===
            ref_tokens = self._forward_sam_ref(ref_sam_featuers, ref_interm_features , ref_prompts)
            # ref_tokens.shape = [batch_size, n_ref, num_mask_outs=5, embedding_dim=256]
            ref_tokens_cat = torch.cat(ref_tokens, dim=0)
            # ref_tokens.shape = [batch_size+sum(n_ref), num_mask_outs=5, embedding_dim=256]
        
        # FINAL FEATURES: corresponding to the target image
        # First B features correspond to the batch inputs.
        # This is currently unused. Used for the second stage (segmentation)
        # TODO: Check HQ SAM if the final features should be used for 1st stage bbox
        sam_features = all_sam_features[:batch_size]

        # INTERMEDIATE FEATURES: corresponding to the target image = batch_inputs
        interm_features_raw = [x[:batch_size] for x in all_interm_features]
        
        # === Rearrange the features ===
        # [num_feature_levels=4, B=3, hidden_dim_channels=768, H_dim=64, W_dim=64]
        interm_features = tuple([x.permute(0, 3, 1, 2) for x in interm_features_raw])
           
        # Get unique labels for target images for each batch
        unique_labels = [torch.unique(batch.gt_instances.labels) for batch in batch_data_samples]
        
        # === Store in cache the precomputed reference tokens ===
        if self.mode == 'compute_embeds':
            # BATCH SIZE should always be 1 when running embedding precomputation
            assert len(batch_data_samples) == 1, "Only support batch size of 1 for embeds precomputation"
            assert len(ref_samples) == 1, "Only support 1 reference image per target image"
            assert len(ref_samples[0].labels) == 1, "Only support 1 label per reference image" 
            self.query_cache[self.ref_index] = ref_tokens_cat[0]
            self.label_cache[self.ref_index] = ref_samples[0].labels[0]
            self.ref_index += 1

            if self.ref_index % 10 == 0:  # Print progress
                print(f"Precomputed {self.ref_index} embeddings")

            if self.ref_index == self.num_categories*self.n_shot:  # Done
                query_cache_np = self.query_cache.cpu().numpy()
                label_cache_np = self.label_cache.cpu().numpy()
                np.save(self.embeds_path, {"query_cache": query_cache_np, "label_cache": label_cache_np})
                print("Saved precomputed embeddings to disk")
                exit()

        if self.mode == 'eval_embeds':
            # Load precomputed embeddings
            # We concat the unique labels along the batch dimension
            unique_labels = torch.cat(unique_labels, dim=0)
            # Init empty tensor to store the loaded reference tokens
            ref_tokens_cat = torch.zeros((len(unique_labels), self.num_masks_outs, self.embed_dims), device=device)
            for i, label in enumerate(unique_labels):
                inds = (self.label_cache == label).squeeze(1)
                selected_ref_embs = self.query_cache[inds][:self.n_shot] # Select the first n_shot embeddings
                ref_tokens_cat[i] = selected_ref_embs.mean(dim=0)  # overwrite ref_tokens_cat with precomputed

        # Get the number of reference images for each sample in the batch
        # This is equivalent to the number of unique labels for each target image
        num_refs = [len(torch.unique(batch.gt_instances.labels)) for batch in batch_data_samples]

        # ref_tokens_cat.shape = [batch_size+sum(n_ref), num_mask_outs=5, embedding_dim=256]
        # Linear layer to fuse all tokens (Single, 3xMulti, HQ) into a single token
        ref_tokens_cat = self.ref_tokens_fuser(ref_tokens_cat)  # [sum(n_ref), C]
        # ref_tokens_cat.shape = [batch_size+sum(n_ref), embedding_dim=256]

        # === Forward intermediate features into the neck ===
        
        # The neck will be responsible for translating the multiple intermediate features for the target_image
        # Reduce number of channels from 768 to 256
        #   in:   [num_feature_levels=4, B=3, hidden_dim_channels=768, H_dim=64, W_dim=64]
        #   out:  [num_feature_levels=4, B=3, hidden_dim_channels=256, H_dim=64, W_dim=64]
        if self.with_neck:
            interm_features = self.neck(interm_features)

        # === Forward target image in DINO style ===

        # forward target image in DINO style
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(interm_features, batch_data_samples)
        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        # arrange and reshape things
        tmp_dec_in, head_inputs_dict = self.pre_decoder(**encoder_outputs_dict)
        decoder_inputs_dict.update(tmp_dec_in)

        query = decoder_inputs_dict['query']  # [B, n_query, C]
        query_pos = decoder_inputs_dict['query_pos']  # [B, n_query, C]
        memory = decoder_inputs_dict['memory']  # [B, n_feature, C]
        memory_mask = decoder_inputs_dict['memory_mask']
        reference_points = decoder_inputs_dict['reference_points']
        spatial_shapes = decoder_inputs_dict['spatial_shapes']
        level_start_index = decoder_inputs_dict['level_start_index']
        valid_ratios = decoder_inputs_dict['valid_ratios']

        # === Create multiple "copies" (pointers) of the same feature maps / queries ===
        # Decouple things according to class labels + stack them batch-wise
        
        expanded_query = []
        expanded_query_pos = []
        expanded_memory = []
        # expanded_memory_mask = []
        expanded_reference_points = []
        expanded_valid_ratios = []

        for b in range(batch_size):
            num_ref = num_refs[b]
            # We only select the query for the given batch, and duplicate it for the num of images.
            # E.g. Suppose b=1, and num_refs[1]=4, then we duplicate the query for the 2nd batch 4 times
            # because we have 4 reference images for that batch
            expanded_query.append(query[b:b + 1, ...].expand(num_ref, -1, -1))
            expanded_query_pos.append(query_pos[b:b + 1, ...].expand(num_ref, -1, -1))
            expanded_memory.append(memory[b:b + 1, ...].expand(num_ref, -1, -1))
            # expanded_memory_mask.append(memory_mask[b:b + 1, ...].repeat(num_ref, 1))
            expanded_reference_points.append(reference_points[b:b + 1, ...].expand(num_ref, -1, -1))
            expanded_valid_ratios.append(valid_ratios[b:b + 1, ...].expand(num_ref, -1, -1))

        # Query comes from query tokens + reference image tokens
        # We take the duplicated query tokens for each batch and concat all of them in dim(0)
        # We loose the batch dimension here when we do torch.cat().
        # shape[0] should match. len(torch.cat(expanded_query,dim=0)) = len(ref_tokens_cat)
        expanded_query = torch.cat(expanded_query, dim=0) + ref_tokens_cat[:, None, :]
        # torch broadcasting is happening here. Suppose the following example:
        #   · torch.cat(expanded_query, dim=0).shape = [8, 300, 256]
        #       8 is the sum(n_ref) when we concat along batch dim, 300 the num of queries, 256 embed_dim
        #   · ref_tokens_cat[:, None, :].shape = [8, 1, 256]
        # Because we have dimension 1 not matching with 300, torch will broadcast the ref_tokens_cat
        # to match the shape of expanded_query.
        # Thus, the same 256 embedding (e.g. 'cat' embedding) will be added to all queries for that ref.
        # Therefore, the 300 queries are searching for 'cats' in the image.
        
        expanded_query_pos = torch.cat(expanded_query_pos, dim=0)
        # Memory = image features
        expanded_memory = torch.cat(expanded_memory, dim=0)
        # expanded_memory_mask = torch.cat(expanded_memory_mask, dim=0)
        # Reference points are for Deformable Attention
        expanded_reference_points = torch.cat(expanded_reference_points, dim=0)
        expanded_valid_ratios = torch.cat(expanded_valid_ratios, dim=0)

        # DETR decoder
        decoder_outputs_dict = self.forward_decoder(
            expanded_query,
            expanded_query_pos,
            expanded_memory,
            memory_mask,
            expanded_reference_points,
            spatial_shapes,
            level_start_index,
            expanded_valid_ratios
        )
        head_inputs_dict.update(decoder_outputs_dict)
        # head_inputs_dict.keys() = ['enc_outputs_class', 'enc_outputs_coord', 'hidden_states', 'references']

        # Decouple batch samples according to the reference images
        # (we duplicate the batch samples according to the number of reference images)
        #
        # [batch_size, DetDataSample]
        decoupled_batch_samples = self._decouple_batch_samples(num_refs, batch_data_samples)

        return head_inputs_dict, decoupled_batch_samples, sam_features, interm_features_raw

    def merge_predictions_into_image(self, batch_data_samples, num_refs, predictions):
        counter = 0
        merged_batch_predictions = []  # store all the joint predictions for each batch
        unique_labels = [torch.unique(batch.gt_instances.labels) for batch in batch_data_samples]
        for i, batch in enumerate(batch_data_samples):  # for each batch
            # Get the n_ref for a given batch
            batch_predictions = predictions[counter:counter + num_refs[i]]
            assert len(batch_predictions) == num_refs[i]
            # assert len(batch.ref_data.ref_data_samples) == len(batch_predictions)
            # Replace label predictions with oracle (= reference image) labels
            for j, label in enumerate(unique_labels[i]):
                batch_predictions[j].labels.fill_(label)
            # Merge into a single instance all the predictions from the reference images
            # (sharing the same target image)
            merged_instance = InstanceData.cat(batch_predictions)
            merged_batch_predictions.append(merged_instance)

            counter += num_refs[i]  # update the counter

        return merged_batch_predictions

    def predict(
        self,
        batch_inputs: Tensor,
        batch_data_samples: OptSampleList, # SampleList
        rescale: bool = True
    ):  
        head_inputs_dict, decoupled_batch_samples, target_sam_feat, tar_interm_embeds = self._forward(batch_inputs, batch_data_samples)

        # predictions = self.bbox_head.predict(
        #     **head_inputs_dict, batch_data_samples=decoupled_batch_samples, rescale=rescale)
        batch_img_metas = [data_samples.metainfo for data_samples in decoupled_batch_samples]
        outs = self.bbox_head(head_inputs_dict['hidden_states'], head_inputs_dict['references'])
        predictions = self.bbox_head.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=False)

        # Filter out the predictions that have a confidence score lower than 0.7
        # new_preds = [pred[pred.scores > 0.7] for pred in predictions]
        # ==> Run SAM prompted with the filtered bboxes to get the segmentation masks
        for i, pred in enumerate(predictions):
            low_res_masks, iou_predictions = self.sam_predict_segm(target_sam_feat, tar_interm_embeds, pred.bboxes)
            masks = self.sam.postprocess_masks(low_res_masks,
                                input_size=batch_data_samples[0].metainfo['img_shape'],
                                original_size=batch_data_samples[0].metainfo['img_shape'])
            # We do MANUAL rescaling of the masks
            original_img_size = batch_data_samples[0].metainfo['ori_shape']
            scale_factor = batch_data_samples[0].metainfo['scale_factor']
            h = int(original_img_size[0] * scale_factor[0] + 0.5)  # 0.5 ensures we round to the nearest integer
            w = int(original_img_size[1] * scale_factor[1] + 0.5)
            unpad_mask = masks[..., :h, :w]
            masks = F.interpolate(unpad_mask, size=original_img_size,
                                       mode='bilinear', align_corners=False).squeeze()
            pred.masks = (masks > self.sam.mask_threshold).squeeze()

        # Scale boxes to ori_shape after SAM
        if rescale:
            for i, pred in enumerate(predictions):
                img_meta = batch_img_metas[i]
                pred.bboxes = pred.bboxes / pred.bboxes.new_tensor(
                    img_meta['scale_factor']).repeat((1, 2))

        # Merge into single image and return
        num_refs = [len(torch.unique(batch.gt_instances.labels)) for batch in batch_data_samples]

        # For visualisation
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples,
            self.merge_predictions_into_image(batch_data_samples, num_refs, predictions)
        )

        # Visualisation
        # self.visualize_predictions(batch_inputs, batch_data_samples)

        return batch_data_samples


    def _sample_prompts(self, samples, only_box=True):
        assert only_box, "Currently only support bounding box as prompt"
        return {'prompt_types':0, 'prompts': samples.bboxes.tensor}

    # def _decouple_batch_samples_old(self, ref_samples, batch_data_samples):
    #     batch_size = len(ref_samples) # Equivalent to len(batch_data_samples)
    #     decoupled_batch_samples = []
    #     for b in range(batch_size):
    #         ref_labels = ref_samples[b].labels # Equivalent to torch.unique(batch_data_samples[b].gt_instances.labels)
    #         _gt_instances = batch_data_samples[b].gt_instances
    #         _ignored_instances = batch_data_samples[b].ignored_instances

    #         for i in range(ref_labels.size(0)):
    #             label_i = ref_labels[i]

    #             new_sample = DetDataSample(metainfo=batch_data_samples[b].metainfo)
    #             new_sample.gt_instances = InstanceData()
    #             new_sample.ignored_instances = InstanceData()

    #             new_sample.gt_instances.labels = torch.zeros_like(
    #                 _gt_instances.labels[_gt_instances.labels == label_i])
    #             new_sample.gt_instances.bboxes = _gt_instances.bboxes[_gt_instances.labels == label_i]

    #             new_sample.ignored_instances.labels = torch.zeros_like(
    #                 _ignored_instances.labels[_ignored_instances.labels == label_i]
    #             )
    #             new_sample.ignored_instances.bboxes = _ignored_instances.bboxes[
    #                 _ignored_instances.labels == label_i
    #             ]
    #             decoupled_batch_samples.append(new_sample)
    #     return decoupled_batch_samples
    
    def _decouple_batch_samples(self, num_refs, batch_data_samples):
        batch_size = len(batch_data_samples) # Equivalent to len(batch_data_samples)
        decoupled_batch_samples = []
        for b in range(batch_size):
            ref_labels = torch.unique(batch_data_samples[b].gt_instances.labels)
            _gt_instances = batch_data_samples[b].gt_instances
            _ignored_instances = batch_data_samples[b].ignored_instances

            for i in range(num_refs[b]):
                label_i = ref_labels[i]

                new_sample = DetDataSample(metainfo=batch_data_samples[b].metainfo)
                new_sample.gt_instances = InstanceData()
                new_sample.ignored_instances = InstanceData()

                new_sample.gt_instances.labels = torch.zeros_like(
                    _gt_instances.labels[_gt_instances.labels == label_i])
                new_sample.gt_instances.bboxes = _gt_instances.bboxes[_gt_instances.labels == label_i]

                new_sample.ignored_instances.labels = torch.zeros_like(
                    _ignored_instances.labels[_ignored_instances.labels == label_i]
                )
                new_sample.ignored_instances.bboxes = _ignored_instances.bboxes[
                    _ignored_instances.labels == label_i
                ]
                decoupled_batch_samples.append(new_sample)
        return decoupled_batch_samples

    def denormalise(self, x, pixel_mean=[123.675, 116.28, 103.53],pixel_std=[58.395, 57.12, 57.375]):
        pixel_mean = np.array(pixel_mean).reshape(3, 1, 1)
        pixel_std = np.array(pixel_std).reshape(3, 1, 1)
        return (x * pixel_std) + pixel_mean

    def merge_images(self, folder_path, output_path):
        from PIL import Image
        import os
        images = []

        # Load all images from the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(folder_path, filename)
                images.append(Image.open(img_path))

        # Calculate the number of rows and columns for the mosaic
        num_images = len(images)
        num_cols = 2
        num_rows = (num_images + num_cols - 1) // num_cols

        # Calculate the size of the mosaic image
        max_width = max(img.width for img in images)
        max_height = max(img.height for img in images)
        mosaic_width = max_width * num_cols
        mosaic_height = max_height * num_rows

        # Create a new blank image for the mosaic
        mosaic = Image.new('RGB', (mosaic_width, mosaic_height), color='white')

        # Paste each image into the mosaic
        for i, img in enumerate(images):
            col_idx = i % num_cols
            row_idx = i // num_cols
            x_offset = col_idx * max_width
            y_offset = row_idx * max_height
            mosaic.paste(img, (x_offset, y_offset))

        # Save the mosaic image
        mosaic.save(output_path)
        print("Mosaic image saved as:", output_path)

    def visualize_predictions(self, batch_inputs, batch_data_samples):
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        # Create a folder to store the visualisation
        visualisation_folder = "results-vis"
        if not os.path.exists(visualisation_folder):
            os.mkdir(visualisation_folder)

        # Create folders
        items = os.listdir(visualisation_folder)
        folders = [name for name in items if os.path.isdir(os.path.join(visualisation_folder, name))]
        folders = [x for x in folders if 'batch' in x]
        next_name = 0
        if len(folders) > 0:
            next_name = max([int(x.split('_')[-1]) for x in folders]) + 1
        visualisation_folder = os.path.join(visualisation_folder, f"batch_{next_name}")
        if not os.path.exists(visualisation_folder):
            os.mkdir(visualisation_folder)
        
        # Visualise the target images with the bboxes and masks overlay
        batch_size = batch_inputs.shape[0] # (batch_size) x 3 x 1024 x 1024
        for i in range(batch_size):
            original_img_size = batch_data_samples[i].metainfo['ori_shape']
            scale_factor = batch_data_samples[i].metainfo['scale_factor']
            # Target image: remove padding and resize to original
            target_img = batch_inputs[i]
            h = int(original_img_size[0] * scale_factor[0] + 0.5)  # 0.5 ensures we round to the nearest integer
            w = int(original_img_size[1] * scale_factor[1] + 0.5)
            unpad_img = target_img[:, :h, :w]
            target_img = F.interpolate(unpad_img.unsqueeze(0), size=original_img_size,
                                       mode='bilinear', align_corners=False).squeeze()
            # Detach and transpose
            target_img = target_img.detach().cpu().numpy().astype(np.uint8).transpose(1,2,0)
            # Change bgr to rgb
            target_img = target_img[..., ::-1]
            plt.figure()
            plt.tight_layout()
            plt.imshow(target_img)
            plt.title("Target Image")

            # Get from GT instances the bboxes and masks.
            # Reshape GT bboxes to original size
            gt_instances = batch_data_samples[i].gt_instances
            bboxes = gt_instances.bboxes
            bboxes = bboxes / bboxes.new_tensor(scale_factor).repeat((1, 2))
            bboxes = bboxes.detach().cpu().numpy()
            # Reshape GT masks to original size
            if hasattr(gt_instances, 'masks'):
                masks = gt_instances.masks.masks
                masks = torch.from_numpy(masks[:, :h, :w].astype(np.float64))
                masks = F.interpolate(masks.unsqueeze(1), size=original_img_size,
                                      mode='bilinear').squeeze().detach().cpu().numpy()
            labels = gt_instances.labels.detach().cpu().numpy()
            
            n_classes = len(np.unique(labels))
            # sample n_classes colors from cmap as rgb values
            colors = plt.cm.get_cmap('tab10', n_classes).colors
            labeltocolor = {lab: color for lab, color in zip(np.unique(labels), colors)}
            
            for j in range(bboxes.shape[0]):
                bbox = bboxes[j]  # plot the bbox overlayed on top of the image
                plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], fill=False, edgecolor=labeltocolor[labels[j]], linewidth=1))
                # plot the mask overlayed on top of the image
                if hasattr(gt_instances, 'masks'):
                    mask = masks[j]
                    mask_rgb = np.expand_dims(mask, axis=-1) * labeltocolor[labels[j]]
                    masked_array = np.ma.masked_where(mask_rgb == 0, mask_rgb)
                    plt.imshow(masked_array, alpha=0.75)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"{visualisation_folder}/target_img_{i}.png")
            plt.close()
            
            
            # ===============================
        
            plt.figure()
            plt.tight_layout()
            plt.imshow(target_img)
            plt.title("Pred Image")
            # get from the predicted instances the bboxes
            pred_instances = batch_data_samples[i].pred_instances
            bboxes = pred_instances.bboxes.detach().cpu().numpy()
            labels = pred_instances.labels.detach().cpu().numpy()
            scores = pred_instances.scores.detach().cpu().numpy()
            threshold_scores = 0.3
            bboxes = bboxes[scores > threshold_scores]
            labels = labels[scores > threshold_scores]
            if hasattr(pred_instances, 'masks'):
                masks = pred_instances.masks.detach().cpu().numpy()
                masks = masks[scores > threshold_scores]
            
            n_classes = len(np.unique(labels))
            # sample n_classes colors from cmap as rgb values
            colors = plt.cm.get_cmap('tab10', n_classes).colors
            
            for j in range(bboxes.shape[0]):
                # plot the bbox overlayed on top of the image
                bbox = bboxes[j]
                plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], fill=False, edgecolor=labeltocolor[labels[j]], linewidth=1))
                # Plot mask
                if hasattr(pred_instances, 'masks'):
                    mask = masks[j]
                    mask_rgb = np.expand_dims(mask, axis=-1) * labeltocolor[labels[j]]
                    masked_array = np.ma.masked_where(mask_rgb == 0, mask_rgb)
                    plt.imshow(masked_array, alpha=0.75)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"{visualisation_folder}/pred_img_{i}.png")
            plt.close()
            # ===============================

            # If ref_data exists
            if hasattr(batch_data_samples[i], 'ref_data'):
                # Ref images visualisation
                ref_imgs = batch_data_samples[i].ref_data.ref_images.detach().cpu().numpy().transpose(0,2,3,1)
                ref_data_samples = batch_data_samples[i].ref_data.ref_data_samples
                
                for j in range(ref_imgs.shape[0]):
                    ref_img = ref_imgs[j].astype(np.uint8)
                    # Change bgr to rgb
                    ref_img = ref_img[..., ::-1]
                    plt.figure()
                    plt.tight_layout()
                    plt.imshow(ref_img)
                    plt.title("Reference Image")
                    # There should only be 1 label, 1 bbox and 1 mask
                    assert len(ref_data_samples[j].labels) == 1
                    assert len(ref_data_samples[j].bboxes) == 1
                    if hasattr(ref_data_samples[j], 'masks'):
                        assert len(ref_data_samples[j].masks.masks) == 1
                    
                    # Plot box
                    labels = ref_data_samples[j].labels.detach().cpu().numpy()
                    bbox = ref_data_samples[j].bboxes[0][0]

                    bbox = bbox.detach().cpu().numpy().squeeze()
                    plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], fill=False, edgecolor=labeltocolor[labels[0]], linewidth=1))
                    # Plot mask
                    if hasattr(ref_data_samples[j], 'masks'):
                        mask = ref_data_samples[j].masks.masks[0]
                        mask_rgb = np.expand_dims(mask, axis=-1) * labeltocolor[labels[0]]
                        masked_array = np.ma.masked_where(mask_rgb == 0, mask_rgb)
                        plt.imshow(masked_array, alpha=0.75)
                    
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(f"{visualisation_folder}/ref_img_{i}_{j}.png")
                    plt.close()
        
        # Merge all images into mosaic
        self.merge_images(visualisation_folder, f"{visualisation_folder}/output_mosaic.png")
        
        
    def plot_batch_img4embeddings(self, batch_inputs, batch_data_samples):
        # === Visualisation of target images, ground trughts and reference images ===
        
        # Create a folder to store the visualisation
        # The title of the image will have the label number
        # The image should have the bbox and the mask overlayed on top of the image
        
        import matplotlib.pyplot as plt
        import numpy as np
        import torchvision
        import torchvision.transforms as T
        from PIL import Image
        import os

        # Create a folder to store the visualisation
        visualisation_folder = "visualisation_shot"
        if not os.path.exists(visualisation_folder):
            os.mkdir(visualisation_folder)
            
        # Create a folder for the label if it does not exist
        label = batch_data_samples[0].gt_instances.labels[0].item()
        visualisation_folder = os.path.join(visualisation_folder, f"label_{label}")
        if not os.path.exists(visualisation_folder):
            os.mkdir(visualisation_folder)
        
        # Find the largest number for the png images
        items = os.listdir(visualisation_folder)
        png_files = [name for name in items if name.endswith('.png')]
        if len(png_files) > 0:
            next_name = max([int(x.split('_')[-1].split('.')[0]) for x in png_files]) + 1
        else:
            next_name = 0
        
        # Visualise the target images with the bboxes and masks overlayed
        batch_size = batch_inputs.shape[0] # (batch_size) x 3 x 1024 x 1024      
        for i in range(batch_size):
            # Detach and denormalise by multiplying variance and adding mean
            target_img = batch_inputs[i].detach().cpu().numpy().astype(np.uint8).transpose(1,2,0)
            # Change bgr to rgb
            target_img = target_img[..., ::-1]
            # target_img = batch_inputs[i].detach().cpu().numpy().transpose(1,2,0)
            # target_img = (target_img * 0.225 + 0.45) * 255
            # target_img = target_img.astype(np.uint8)
            plt.figure()
            plt.imshow(target_img)
            plt.title("Target Image")
            # get from the ground truth instances the bboxes and masks
            gt_instances = batch_data_samples[i].gt_instances
            bboxes = gt_instances.bboxes.detach().cpu().numpy()
            masks = gt_instances.masks
            labels = gt_instances.labels.detach().cpu().numpy()
            
            n_classes = len(np.unique(labels))
            # sample n_classes colors from cmap as rgb values
            colors = plt.cm.get_cmap('tab10', n_classes).colors
            labeltocolor = {lab: color for lab, color in zip(np.unique(labels), colors)}
            
            for j in range(bboxes.shape[0]):
                # plot the bbox overlayed on top of the image
                bbox = bboxes[j]
                plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], fill=False, edgecolor='red', linewidth=2))
                # plot the mask overlayed on top of the image
                mask = masks.masks[j]
                mask_rgb = np.expand_dims(mask, axis=-1) * labeltocolor[labels[j]]
                masked_array = np.ma.masked_where(mask_rgb == 0, mask_rgb)
                plt.imshow(masked_array, alpha=0.75)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"{visualisation_folder}/target_img_{next_name}.png")
            plt.close()
            
            
            # Ref images visualisation
            ref_imgs = batch_data_samples[i].ref_data.ref_images.detach().cpu().numpy().transpose(0,2,3,1)
            ref_data_samples = batch_data_samples[i].ref_data.ref_data_samples
            
            for j in range(ref_imgs.shape[0]):
                ref_img = ref_imgs[j].astype(np.uint8)
                # Change bgr to rgb
                ref_img = ref_img[..., ::-1]
                plt.figure()
                plt.imshow(ref_img)
                plt.title("Reference Image")
                # There should only be 1 label, 1 bbox and 1 mask
                assert len(ref_data_samples[j].labels) == 1
                assert len(ref_data_samples[j].bboxes) == 1
                assert len(ref_data_samples[j].masks.masks) == 1
                
                # Plot box
                bbox = ref_data_samples[j].bboxes[0][0].detach().cpu().numpy().squeeze()
                # Plot mask
                mask = ref_data_samples[j].masks.masks[0]
                labels = ref_data_samples[j].labels.detach().cpu().numpy()
                mask_rgb = np.expand_dims(mask, axis=-1) * labeltocolor[labels[0]]
                masked_array = np.ma.masked_where(mask_rgb == 0, mask_rgb)
                plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], fill=False, edgecolor='red', linewidth=2))
                plt.imshow(masked_array, alpha=0.75)
                
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(f"{visualisation_folder}/ref_img_{next_name}_{j}.png")
                plt.close()
            
        print(f" ==> Plot chosen images for label {label}.")
        # exit()
        
    def plot_batch(self, batch_inputs, batch_data_samples):
        # === Visualisation of target images, ground trughts and reference images ===
        
        # Create a folder to store the visualisation
        # The title of the image will have the label number
        # The image should have the bbox and the mask overlayed on top of the image
        
        import matplotlib.pyplot as plt
        import numpy as np
        import torchvision
        import torchvision.transforms as T
        from PIL import Image
        import os

        # Create a folder to store the visualisation
        visualisation_folder = "visualisation"
        if not os.path.exists(visualisation_folder):
            os.mkdir(visualisation_folder)
            
        # Visualise the target images with the bboxes and masks overlayed
        batch_size = batch_inputs.shape[0] # (batch_size) x 3 x 1024 x 1024      
        for i in range(batch_size):
            # Detach and denormalise by multiplying variance and adding mean
            target_img = batch_inputs[i].detach().cpu().numpy().astype(np.uint8).transpose(1,2,0)
            # Change bgr to rgb
            target_img = target_img[..., ::-1]
            # target_img = batch_inputs[i].detach().cpu().numpy().transpose(1,2,0)
            # target_img = (target_img * 0.225 + 0.45) * 255
            # target_img = target_img.astype(np.uint8)
            plt.figure()
            plt.imshow(target_img)
            plt.title("Target Image")
            # get from the ground truth instances the bboxes and masks
            gt_instances = batch_data_samples[i].gt_instances
            bboxes = gt_instances.bboxes.detach().cpu().numpy()
            # only assign if gt_instances has masks attribute
            if hasattr(gt_instances, 'masks'):
                masks = gt_instances.masks
            labels = gt_instances.labels.detach().cpu().numpy()
            
            n_classes = len(np.unique(labels))
            # sample n_classes colors from cmap as rgb values
            colors = plt.cm.get_cmap('tab10', n_classes).colors
            labeltocolor = {lab: color for lab, color in zip(np.unique(labels), colors)}
            
            for j in range(bboxes.shape[0]):
                # plot the bbox overlayed on top of the image
                bbox = bboxes[j]
                plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], fill=False, edgecolor='red', linewidth=2))
                # plot the mask overlayed on top of the image
                if hasattr(gt_instances, 'masks'):
                    mask = masks.masks[j]
                    mask_rgb = np.expand_dims(mask, axis=-1) * labeltocolor[labels[j]]
                    masked_array = np.ma.masked_where(mask_rgb == 0, mask_rgb)
                    plt.imshow(masked_array, alpha=0.75)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"{visualisation_folder}/target_img_{i}.png")
            
            
            # Ref images visualisation
            ref_imgs = batch_data_samples[i].ref_data.ref_images.detach().cpu().numpy().transpose(0,2,3,1)
            ref_data_samples = batch_data_samples[i].ref_data.ref_data_samples
            
            for j in range(ref_imgs.shape[0]):
                ref_img = ref_imgs[j].astype(np.uint8)
                # Change bgr to rgb
                ref_img = ref_img[..., ::-1]
                plt.figure()
                plt.imshow(ref_img)
                plt.title("Reference Image")
                # There should only be 1 label, 1 bbox and 1 mask
                assert len(ref_data_samples[j].labels) == 1
                assert len(ref_data_samples[j].bboxes) == 1
                if hasattr(ref_data_samples[j], 'masks'):
                    assert len(ref_data_samples[j].masks.masks) == 1
                
                # Plot box
                bbox = ref_data_samples[j].bboxes[0][0].detach().cpu().numpy().squeeze()
                plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], fill=False, edgecolor='red', linewidth=2))
                # Plot mask
                labels = ref_data_samples[j].labels.detach().cpu().numpy()
                if hasattr(ref_data_samples[j], 'masks'):
                    mask = ref_data_samples[j].masks.masks[0]
                    mask_rgb = np.expand_dims(mask, axis=-1) * labeltocolor[labels[0]]
                    masked_array = np.ma.masked_where(mask_rgb == 0, mask_rgb)
                    plt.imshow(masked_array, alpha=0.75)

                plt.axis('off')
                plt.tight_layout()
                plt.savefig(f"{visualisation_folder}/ref_img_{i}_{j}.png")
            
        print(" ===> We only plot the first batch of images.")
        exit()

    def loss(
            self,
            batch_inputs: Tensor, # targets
            batch_data_samples: OptSampleList = None
    ):
        head_inputs_dict, decoupled_batch_samples, _, _ = self._forward(batch_inputs, batch_data_samples)
        
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=decoupled_batch_samples)
        return losses

    def forward_decoder(
        self,
        query: Tensor,
        query_pos: Tensor,
        memory: Tensor,
        memory_mask: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor
    ):
        inter_states, inter_references = self.decoder(
            query=query,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=memory_mask,  # for cross_attn
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches
            if self.with_box_refine else None)
        references = [reference_points, *inter_references]

        decoder_outputs_dict = dict(
            hidden_states=inter_states, references=references)
        return decoder_outputs_dict

    @torch.no_grad()
    def _extract_sam_features(self, images):
        self.sam.eval()
        imgs_normed = self.sam.preprocess(images)
        final_features, interm_features = self.sam.image_encoder(imgs_normed)
        return final_features, interm_features

    @torch.no_grad()
    def _forward_sam_ref(self, ref_final_images, ref_interm_features, ref_prompts):
        self.sam.eval()
        num_ref = [x['prompts'].shape[0] for x in ref_prompts]
        batch_size = len(num_ref)

        # imgs = torch.cat(ref_images, dim=0)
        # imgs_normed = self.sam.preprocess(imgs)
        #
        # final_features, interm_features = self.sam.image_encoder(imgs_normed)
        final_features, interm_features = self._split_features(num_ref, ref_final_images, ref_interm_features)

        outputs_all_batch = []

        for b in range(batch_size):
            b_num_ref = num_ref[b]
            b_features = final_features[b] # .expand(b_num_ref, -1, -1, -1)
            b_interm_features = interm_features[b][0] # [None, ...].expand(b_num_ref, -1, -1, -1)

            b_prompts_type = ref_prompts[b]['prompt_types']
            b_prompts = ref_prompts[b]['prompts']

            if b_prompts_type == 0:  # bbox
                curr_prompt = b_prompts.reshape(b_num_ref, -1, 4)
                sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                    points=None, boxes=curr_prompt, masks=None,
                )
            elif b_prompts_type == 1:  # points
                raise NotImplementedError
                sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                    points=curr_prompt, boxes=None, masks=None,
                )
            elif b_prompts_type == 2:  # masks
                raise NotImplementedError
                sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                    points=None, boxes=None, masks=curr_prompt,
                )
            else:
                raise NotImplementedError
            hs_tokens = self.sam.mask_decoder.token_interact(
                image_embeddings=b_features,
                image_pe=self.sam.prompt_encoder.get_dense_pe().expand((b_num_ref, -1, -1, -1)),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                expand_src=False
            )

            outputs_all_batch.append(hs_tokens)
        return outputs_all_batch

    def _split_features(self, num_ref, final_features, interm_features):
        splited_final_features = []
        splited_interm_feaures = []
        for i in range(len(num_ref)):
            splited_final_features.append(final_features[sum(num_ref[:i]):sum(num_ref[:(i + 1)])])
            splited_interm_feaures.append([x[sum(num_ref[:i]):sum(num_ref[:(i + 1)])] for x in interm_features])
        return splited_final_features, splited_interm_feaures

    def pre_transformer(
        self,
        mlvl_feats: Tuple[Tensor], # multi-level features
        batch_data_samples: OptSampleList = None
    ):
        batch_size = mlvl_feats[0].size(0)

        # === Construct Masks and Positional Embeddings ===
        
        # construct binary masks for the transformer.
        assert batch_data_samples is not None
        batch_input_shape = batch_data_samples[0].batch_input_shape
        input_img_h, input_img_w = batch_input_shape
        img_shape_list = [sample.img_shape for sample in batch_data_samples]
        same_shape_flag = all([
            s[0] == input_img_h and s[1] == input_img_w for s in img_shape_list
        ])
        # support torch2onnx without feeding masks
        if torch.onnx.is_in_onnx_export() or same_shape_flag:
            mlvl_masks = []
            mlvl_pos_embeds = []
            for feat in mlvl_feats:
                mlvl_masks.append(None)
                mlvl_pos_embeds.append(
                    self.positional_encoding(None, input=feat))
        else:
            masks = mlvl_feats[0].new_ones(
                (batch_size, input_img_h, input_img_w))
            for img_id in range(batch_size):
                img_h, img_w = img_shape_list[img_id]
                masks[img_id, :img_h, :img_w] = 0
            # NOTE following the official DETR repo, non-zero
            # values representing ignored positions, while
            # zero values means valid positions.

            mlvl_masks = []
            mlvl_pos_embeds = []
            for feat in mlvl_feats:
                mlvl_masks.append(
                    F.interpolate(masks[None], size=feat.shape[-2:]).to(
                        torch.bool).squeeze(0))
                mlvl_pos_embeds.append(
                    self.positional_encoding(mlvl_masks[-1]))
        
        # === Flatten features and masks ===
        # from [batch_size, channels, height, width] to [batch_size, height*width, channels]
        feat_flatten = []
        lvl_pos_embed_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            batch_size, c, h, w = feat.shape
            spatial_shape = torch._shape_as_tensor(feat)[2:].to(feat.device)
            # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            pos_embed = pos_embed.view(batch_size, c, -1).permute(0, 2, 1)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            # [bs, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl]
            if mask is not None:
                mask = mask.flatten(1)

            feat_flatten.append(feat)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            mask_flatten.append(mask)
            spatial_shapes.append(spatial_shape)

        # (bs, num_feat_points, dim)
        feat_flatten = torch.cat(feat_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        # (bs, num_feat_points), where num_feat_points = sum_lvl(h_lvl*w_lvl)
        if mask_flatten[0] is not None:
            mask_flatten = torch.cat(mask_flatten, 1)
        else:
            mask_flatten = None

        # (num_level, 2)
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1,)),  # (num_level)
            spatial_shapes.prod(1).cumsum(0)[:-1]))
        if mlvl_masks[0] is not None:
            valid_ratios = torch.stack(  # (bs, num_level, 2)
                [self.get_valid_ratio(m) for m in mlvl_masks], 1)
        else:
            valid_ratios = mlvl_feats[0].new_ones(batch_size, len(mlvl_feats),
                                                  2)

        # Prepare inputs for encoder and decoder
        encoder_inputs_dict = dict(
            feat=feat_flatten,
            feat_mask=mask_flatten,
            feat_pos=lvl_pos_embed_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        decoder_inputs_dict = dict(
            memory_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        return encoder_inputs_dict, decoder_inputs_dict

    def forward_encoder(
        self,
        feat: Tensor,
        feat_mask: Tensor,
        feat_pos: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor
    ):
        memory = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes)
        return encoder_outputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor
    ):
        batch_size, _, c = memory.shape
        enc_outputs_class, enc_outputs_coord = None, None
        query_embed = self.query_embedding.weight
        query_pos, query = torch.split(query_embed, c, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(batch_size, -1, -1)
        query = query.unsqueeze(0).expand(batch_size, -1, -1)
        reference_points = self.reference_points_fc(query_pos).sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            query_pos=query_pos,
            memory=memory,
            reference_points=reference_points)
        head_inputs_dict = dict(
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord=enc_outputs_coord) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict

    def extract_feat(self, batch_inputs: Tensor):
        pass

    def sam_predict_segm(self, target_img_embeds, tar_interm_embeds,
                         bboxes, multimask_output=False):

        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=bboxes,
            masks=None,
        )
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=target_img_embeds,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            interm_embeddings=tar_interm_embeds,
            hq_token_only=False, # Not implemented
        )

        return low_res_masks, iou_predictions



        # return masks, iou_predictions
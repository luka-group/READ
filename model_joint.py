from typing import Optional, Tuple, Union
import copy

import torch.utils.checkpoint
from torch.nn import functional as F

from transformers.modeling_outputs import (
    SequenceClassifierOutput
)
from transformers import BertPreTrainedModel, BertForSequenceClassification

from model import BertForSequenceClassification as MainBertForSequenceClassification


def bias_product_loss(main_logits, bias_logits, labels):
    main_logits = main_logits.float()  # In case we were in fp16 mode
    main_logits = F.log_softmax(main_logits, 1)
    bias_logits = bias_logits.float()  # In case we were in fp16 mode
    bias_logits = F.log_softmax(bias_logits, 1)
    return F.cross_entropy(main_logits + bias_logits, labels)


def bias_add_loss(main_logits, bias_logits, labels, ensemble_ratio=0.1):
    main_probs = F.softmax(main_logits, 1)
    bias_probs = F.softmax(bias_logits, 1)
    ensemble_probs = (1 - ensemble_ratio) * main_probs + ensemble_ratio * bias_probs
    return F.nll_loss(torch.log(ensemble_probs), labels)


class EnsembleBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        if self.config.model_stage == "train_ensemble":
            self.main_model = None
            self.bias_model = None
        elif self.config.model_stage == "eval_main":
            bias_config = copy.deepcopy(config)
            bias_config.num_hidden_layers = config.ensemble_layer_num

            self.main_model = MainBertForSequenceClassification(config)
            self.bias_model = BertForSequenceClassification(bias_config)

    def check_weight(self, i):
        print("main_bias", self.main_model.bert.encoder.layer[i].attention.self.query_bias.weight[0][:10])
        print("main", self.main_model.bert.encoder.layer[i].attention.self.query.weight[0][:10])
        print("bias", self.bias_model.bert.encoder.layer[i].attention.self.query.weight[0][:10])

    def tie_main_and_bias_model(self):
        # self.check_weight(0)

        self.bias_model.bert.embeddings = self.main_model.bert.embeddings
        for i in range(self.config.ensemble_layer_num):
            # share main weights to bias model
            self.bias_model.bert.encoder.layer[i].intermediate = self.main_model.bert.encoder.layer[i].intermediate
            self.bias_model.bert.encoder.layer[i].output = self.main_model.bert.encoder.layer[i].output
            self.bias_model.bert.encoder.layer[i].attention.self.value = self.main_model.bert.encoder.layer[
                i].attention.self.value
            self.bias_model.bert.encoder.layer[i].attention.output = self.main_model.bert.encoder.layer[
                i].attention.output

            # share bias weights to main model
            self.main_model.bert.encoder.layer[i].attention.self.query_bias = self.bias_model.bert.encoder.layer[
                i].attention.self.query
            self.main_model.bert.encoder.layer[i].attention.self.key_bias = self.bias_model.bert.encoder.layer[
                i].attention.self.key

        # self.check_weight(0)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        bias_outputs = self.bias_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        main_outputs = self.main_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        loss = None
        if labels is not None:
            main_loss = bias_product_loss(main_outputs.logits.view(-1, self.num_labels),
                                          bias_outputs.logits.detach().view(-1, self.num_labels),
                                          labels.view(-1))
            loss = main_loss + bias_outputs.loss

        if not return_dict:
            output = (main_outputs.logits,) + main_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=main_outputs.logits,
            hidden_states=main_outputs.hidden_states,
            attentions=main_outputs.attentions,
        )

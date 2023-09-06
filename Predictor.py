import torch
from torch import nn


class IHMP_Predictor(nn.Module):
    def __init__(self, core_model, label_dim):
        super(IHMP_Predictor, self).__init__()
        self.core_model = core_model
        self.hidden_size = core_model.output_dim
        self.label_dim = label_dim
        self.linear = nn.Linear(self.hidden_size, self.label_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        _, final_output = self.core_model(inputs)
        final_output = self.linear(final_output)
        final_output = self.dropout(final_output)
        final_output = self.sigmoid(final_output)
        return final_output


class DP_Predictor(nn.Module):
    def __init__(self, core_model, label_dim):
        super(DP_Predictor, self).__init__()
        self.core_model = core_model
        self.hidden_size = core_model.output_dim
        self.label_dim = label_dim
        self.linear = nn.Linear(self.hidden_size, self.label_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        final_output, _ = self.core_model(inputs)
        final_output = self.linear(final_output)
        final_output = self.dropout(final_output)
        final_output = self.sigmoid(final_output)
        return final_output


class PHEN_Predictor(nn.Module):
    def __init__(self, core_model, label_dim):
        super(PHEN_Predictor, self).__init__()
        self.core_model = core_model
        self.hidden_size = core_model.output_dim
        self.label_dim = label_dim
        self.linear = nn.Linear(self.hidden_size, self.label_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        _, final_output = self.core_model(inputs)
        final_output = self.linear(final_output)
        final_output = self.dropout(final_output)
        final_output = self.sigmoid(final_output)
        return final_output


class LOS_Predictor(nn.Module):
    def __init__(self, core_model, label_dim):
        super(LOS_Predictor, self).__init__()
        self.core_model = core_model
        self.hidden_size = core_model.output_dim
        self.label_dim = label_dim
        self.linear = nn.Linear(self.hidden_size, self.label_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax()

    def forward(self, inputs):
        final_output, _ = self.core_model(inputs)
        final_output = self.linear(final_output)
        final_output = self.dropout(final_output)
        final_output = self.softmax(final_output)
        return final_output



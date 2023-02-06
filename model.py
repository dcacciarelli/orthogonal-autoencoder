from torch import nn


##############################
#                            #
#           Model            #
#                            #
##############################

class OrthogonalAutoEncoder(nn.Module):

    def __init__(self, encoding_layers, activation_function=nn.ReLU(), dropout=0.01):

        # Input dimension is the first item of the list
        # Encoding dimension is the last item of the layers list (decoder is symmetric)

        super().__init__()

        # Encoding layers
        encoding = []
        for i in range(len(encoding_layers) - 1):
            encoding.append(nn.Linear(encoding_layers[i], encoding_layers[i + 1]))
            encoding.append(nn.Dropout(dropout))
            encoding.append(activation_function)
            encoding.append(nn.BatchNorm1d(encoding_layers[i + 1]))

        # Decoding layers
        decoding_layers = encoding_layers[::-1]
        decoding = []
        for i in range(len(decoding_layers) - 1):
            decoding.append(nn.Linear(decoding_layers[i], decoding_layers[i + 1]))
            decoding.append(nn.Dropout(dropout))
            decoding.append(nn.BatchNorm1d(decoding_layers[i + 1]))
            decoding.append(activation_function)

        self.encoder = nn.Sequential(*encoding)
        self.decoder = nn.Sequential(*decoding)

    def forward(self, x):
        encoded_features_all_layers = []  # storing encoded features given in output from each encoding layer
        for layer in self.encoder:
            x = layer(x)
            encoded_features_all_layers.append(x)

        x_compressed = encoded_features_all_layers[-1]
        x_reconstructed = self.decoder(x_compressed)
        return x_compressed, x_reconstructed, encoded_features_all_layers

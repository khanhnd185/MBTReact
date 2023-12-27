import torch
import torch.nn as nn
from .wav2vec2focctc import Wav2Vec2ForCTC
from .utils import  init_biased_mask, enc_dec_mask, PeriodicPositionalEncoding
from torch.nn.modules.transformer import _get_clones

class MBT(nn.Module):
    def __init__(self, dim, num_layers, num_heads, num_bottle_token, device):
        super(MBT, self).__init__()
        self.dim = dim
        self.device = device
        self.num_layers = num_layers
        self.num_bottle_token = num_bottle_token
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads)

        self.x_layers = _get_clones(encoder_layer, num_layers)
        self.y_layers = _get_clones(encoder_layer, num_layers)

        self.bot = nn.Parameter(torch.randn(1, num_bottle_token, dim))

    def get_mask(self, b, l):
        return torch.zeros(b, l+self.num_bottle_token).to(device=self.device)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        mask_x = self.get_mask(x.shape[0], x.shape[1])
        mask_y = self.get_mask(y.shape[0], y.shape[1])

        bot = self.bot.expand(x.shape[0], -1, -1)
        x = torch.cat((bot, x), dim=1)
        y = torch.cat((bot, y), dim=1)

        x = x.permute(1, 0, 2)
        y = y.permute(1, 0, 2)

        for i in range(self.num_layers):
            x = self.x_layers[i](src=x, src_key_padding_mask=mask_x)
            y = self.y_layers[i](src=y, src_key_padding_mask=mask_y)

            x[:self.num_bottle_token] = (x[:self.num_bottle_token] + y[:self.num_bottle_token]) / 2
            y[:self.num_bottle_token] = x[:self.num_bottle_token]

        x = x[self.num_bottle_token:,:,:].permute(1, 0, 2)
        y = y[self.num_bottle_token:,:,:].permute(1, 0, 2)

        return x, y


class SpeakFormer(nn.Module):
    def __init__(self, img_size=224, feature_dim = 256, period = 25, max_seq_len = 751,  device = 'cpu', use_mbt=True):
        super(SpeakFormer, self).__init__()

        self.use_mbt = use_mbt
        self.img_size = img_size

        self.feature_dim = feature_dim

        # wav2vec 2.0 weights initialization
        self.audio_encoder = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_encoder.freeze_feature_extractor()
        self.audio_feature_map = nn.Linear(768, feature_dim)


        self.PPE = PeriodicPositionalEncoding(feature_dim, period=period, max_seq_len=max_seq_len)
        self.biased_mask = init_biased_mask(n_head=8, max_seq_len=max_seq_len, period=period)


        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=8, dim_feedforward=2*feature_dim, batch_first=True)
        self.speaker_transformer_decoder1 = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.speaker_transformer_decoder2 = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.speaker_transformer_decoder3 = nn.TransformerDecoder(decoder_layer, num_layers=1)

        if self.use_mbt:
            self.speaker_transformer_decoder3 = MBT(feature_dim, 2, 4, 4, device)
        else:
            self.speaker_transformer_decoder3 = nn.TransformerDecoder(decoder_layer, num_layers=1)

        self.device = device


    def forward(self, video_features, audio):

        frame_num = video_features.shape[1]

        hidden_states = self.audio_encoder(audio, frame_num=frame_num)

        if hidden_states.shape[1]<frame_num*2:
            video_features = video_features[:, : hidden_states.shape[1]//2]
            frame_num = hidden_states.shape[1]//2

        hidden_states = self.audio_feature_map(hidden_states)

        video_features = self.PPE(video_features)
        tgt_mask = self.biased_mask[:, :video_features.shape[1], :video_features.shape[1]].clone().detach().to(device=self.device).repeat(video_features.shape[0],1,1)
        memory_mask = enc_dec_mask(self.device, video_features.shape[1], hidden_states.shape[1])

        speaker_vector = self.speaker_transformer_decoder1(video_features, video_features, tgt_mask=tgt_mask)
        speaker_vector = self.speaker_transformer_decoder2(speaker_vector, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
        if self.use_mbt:
            speaker_motion, hidden_states = self.speaker_transformer_decoder3(speaker_vector, hidden_states)
        else:
            speaker_motion = self.speaker_transformer_decoder3(speaker_vector, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)

        return  speaker_motion, hidden_states, speaker_vector





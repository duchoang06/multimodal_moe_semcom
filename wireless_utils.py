import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools
   

class SimpleWirelessChannel(nn.Module):
    def __init__(self, snr_dB=15, fading='none', rician_k=4.0):
        super().__init__()
        self.snr_dB = snr_dB
        self.fading = fading.lower()
        self.rician_k = rician_k  # Ratio of LOS to scattered power

    def forward(self, x, snr, fading='none', rician_k=3.0, modal=False): # x in [batch, seq_len, dim]
        snr_linear = 10 ** (snr / 10)
        noise_std = math.sqrt(1 / snr_linear)

        if fading == 'rayleigh':
            h = torch.randn_like(x) 

        elif fading == 'rician':
            # Rician fading: LOS + Rayleigh
            los = torch.ones_like(x)  # deterministic LOS component
            rayleigh = torch.randn_like(x)
            factor = math.sqrt(rician_k / (rician_k + 1))
            scatter = math.sqrt(1 / (rician_k + 1))
            h = factor * los + scatter * rayleigh

        elif fading == 'none':
            h = torch.ones_like(x)  # No fading = gain of 1

        else:
            raise ValueError(f'Unknown fading type: {fading}')

        x_faded = x * h
        noise = noise_std * torch.randn_like(x)
        y = x_faded + noise
        y_equalized = y / h

        h_codeword = torch.mean(h**2, dim=-1) 

        snr_fading_arr = 10 * torch.log10( h_codeword/ (noise_std ** 2) )

        return y_equalized, snr_fading_arr.numpy()

class ComplexWirelessChannel(nn.Module):
    def __init__(self, snr_dB=15, fading='none', rician_k=4.0):
        super().__init__()
        self.snr_dB = snr_dB
        self.fading = fading.lower()
        self.rician_k = rician_k

    def power_norm_batchwise(self, signal, power=1.0):
        # batch_size, num_elements = signal.shape[0], len(signal[0].flatten())
        # num_complex = num_elements // 2
        # signal_shape = signal.shape
        # signal = signal.view(batch_size, num_complex, 2)
        num_complex = signal.shape[-1]

        signal_power = torch.sum((signal[:,:,0]**2 + signal[:,:,1]**2), dim=-1) / num_complex
        signal = signal * math.sqrt(power) / torch.sqrt(signal_power.unsqueeze(-1).unsqueeze(-1))

        return signal

    def apply_fading(self, signal, fading_type, rician_k=4.0):
        batch_size, num_symbols, _ = signal.shape
        device = signal.device

        if fading_type == 'rayleigh':
            h_real = torch.normal(0, math.sqrt(1/2), size=[batch_size, 1]).to(device)
            h_imag = torch.normal(0, math.sqrt(1/2), size=[batch_size, 1]).to(device)
        elif fading_type == 'rician':
            mean = math.sqrt(rician_k / (rician_k + 1))
            std = math.sqrt(1 / (rician_k + 1))
            h_real = torch.normal(mean, std, size=[batch_size, 1]).to(device)
            h_imag = torch.normal(mean, std, size=[batch_size, 1]).to(device)
        elif fading_type == 'none':
            h_real = torch.ones(batch_size, 1).to(device)
            h_imag = torch.zeros(batch_size, 1).to(device)
        else:
            raise ValueError(f"Unknown fading type: {fading_type}")

        h_real = h_real.squeeze(1)  
        h_imag = h_imag.squeeze(1)  

        H = torch.stack([
            torch.stack([h_real, -h_imag], dim=-1),
            torch.stack([h_imag,  h_real], dim=-1)
        ], dim=-2)  # (batch, 2, 2)

        faded_signal = torch.matmul(signal, H)  # (batch, num_symbols, 2)

        return faded_signal, H

    def forward(self, x, snr=None, fading=None, rician_k=4.0, modal=False, noisy_k=1.0):
        """
        x: [batch, seq_len, dim] real-valued input
        """
        batch_size = x.shape[0]
        snr = self.snr_dB if snr is None else snr
        fading = self.fading if fading is None else fading.lower()

        # Flatten and group into complex pairs
        x_flat = x.view(batch_size, -1) # (batch, seq_len * dim)
        original_shape = x.shape

        # Pad if necessary
        padded = False
        if x_flat.shape[-1] % 2 != 0:
            x_flat = torch.cat([x_flat, torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)], dim=-1)
            padded = True
        else:
            padded = False
        
        num_symbols = x_flat.shape[-1] // 2
        x_complex = x_flat.view(batch_size, num_symbols, 2)

        # Normalize power
        x_complex = self.power_norm_batchwise(x_complex, power=1.0)

        # Apply fading
        y_complex, H = self.apply_fading(x_complex, fading_type=fading, rician_k=rician_k)
        H_inv = torch.linalg.inv(H)  

        # Compute noise power
        snr_linear = 10 ** (snr / 10)
        
        if modal:
            noise_std = (0.95 + noisy_k/15)*math.sqrt(1 / (2*snr_linear) ) # previously 1.25 #1.1 for weight ablation , 0.7
        else:
            noise_std = math.sqrt(1 / (2*snr_linear) )

        # noise_std = math.sqrt(1 / (2*snr_linear) )

        # Add AWGN
        noise = noise_std * torch.randn_like(y_complex)

        y_noisy = y_complex + noise

        y_equalized = torch.matmul(y_noisy, H_inv)

        # Reshape back to (batch, seq_len, dim)
        y_flat = y_equalized.view(batch_size, -1)
        if padded:
            y_flat = y_flat[:, :-1]

        y_out = y_flat.view(original_shape)

        return y_out, x_complex.view(original_shape), y_noisy.view(original_shape)
    
def mi_loss_channel(tx_signal, rx_signal, critic):
    # critic is a neural network
    joint, marginal = sample_batch(tx_signal, rx_signal)

    t = critic(joint)
    et = torch.exp(critic(marginal)) 

    mi_loss = torch.mean(t) - torch.log(torch.mean(et) + 1e-8)

    return -mi_loss

def sample_batch(rec, noise):
    rec = torch.reshape(rec, shape=(-1, 1))
    noise = torch.reshape(noise, shape=(-1, 1))
    rec_sample1, rec_sample2 = torch.split(rec, int(rec.shape[0]/2), dim=0)
    noise_sample1, noise_sample2 = torch.split(noise, int(noise.shape[0]/2), dim=0)
    joint = torch.cat((rec_sample1, noise_sample1), 1)
    marg = torch.cat((rec_sample1, noise_sample2), 1)
    return joint, marg
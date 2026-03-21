# MaxMark
This is the official implementation of MaxMark: High-Capacity Diffusion-Native Watermarking via Robust and Invertible Latent Embedding, accepted by CVPR 2026.
# Method

![fig_flow_01](.assets/fig_flow_01.png)

We propose MaxMark, a latent-based watermarking method that enables high-capacity and reliable watermark extraction while preserving image quality. Achieving high capacity requires three properties: (1) a robust embedding strategy that places information in reliable latent regions, (2) a mechanism to map perturbed latents back to the LDM’s native Gaussian prior, and (3) minimal loss in both embedding and extraction to ensure accurate recovery. MaxMark satisfies these requirements through a robust watermark embedding module, which enhances and embeds the watermark payload, and a distribution transformation module, which uses an invertible neural network (INN) to map the watermarked latent back to the standard Gaussian prior. These modules are designed by our observations that sign bits serve as reliable information carriers, ECC parameters can be automatically tuned for stability, and invertibility is critical for minimizing recovery loss.

# Setup

```shell
pip install -r requirements.txt
```

# Run

#### training

For the SD v1.5, v2.0, etc. The latent shape is set as 4x64x64, if you train INN for other LDMs, please check the dimensions of latent space. And the secret length should correspond to the shape of the latent space.

```shell
python train_INN.py --secret_length 16384 \
	--threshold 0.1 \
	--epochs 400 \
    --margin 10.0
```

#### evaluation

Embedding the watermark secret with ECC method like Reed–Solomon, and evaluate the bit accuracy at different secret length.

```sh
python evaluation-rs-attack.py --model_path YOUR_LDM_PATH \
	--secret_length 1024 \
	--model THE_INN_PATH \
	--margin 10.0 \
	--total_size 16384 \
	--ecc_backups 5 \
	--data_backups 3 \
	--save SAVE_PATH
```

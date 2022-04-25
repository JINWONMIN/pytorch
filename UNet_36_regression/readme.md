## Denoising

<pre>
<code>
python3 train.py \
        --mode train \
        --network unet \
        --learning_type residual\
        --task denoising \
        --opts random 30.0
</code>
</pre>
---
<pre>
<code>
python3 train.py \
        --mode train \
        --network unet \
        --learning_type plain\
        --task denoising \
        --opts random 30.0
</code>
</pre>

## Inpainting

<pre>
<code>
python3 train.py \
        --mode train \
        --learning_type residual \
        --task inpainting \
        --opts uniform 0.5
</code>
</pre>
---
<pre>
<code>
python3 train.py \
        --mode train \
        --learning_type plain \
        --task inpainting \
        --opts uniform 0.5
</code>
</pre>

## Super resolution

<pre>
<code>
python3 train.py \
        --mode train \
        --learning_type plain \
        --task super_resolution \
        --opts bilinear 4.0
</code>
</pre>
---
<pre>
<code>
python3 train.py \
        --mode train \
        --learning_type residual \
        --task super_resolution \
        --opts bilinear 4.0
</code>
</pre>


---
## Reference

<https://www.youtube.com/watch?v=XNE5Up5pCgE>

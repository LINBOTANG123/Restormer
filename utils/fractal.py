import os
import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import random

# Sampling & detail parameters
TEST_ITERS = 500
TEST_LOWER = 20
TEST_UPPER = 95
ZOOM_RANGE = (0.001, 0.01)    # moderate zooms so patterns fill more of frame
ITER_RANGE = (600, 1100)      # iteration count for rendering
DETAIL_THRESHOLD = 0.90       # require ≥90% of pixels > 10

def sample_center():
    """Pick a point near the Mandelbrot boundary."""
    while True:
        c = complex(random.uniform(-2.0, 1.0),
                    random.uniform(-1.5, 1.5))
        z = 0+0j
        for i in range(TEST_ITERS):
            z = z*z + c
            if abs(z) > 2:
                break
        if TEST_LOWER < i < TEST_UPPER:
            return c

def generate_and_save(idx: int, output_dir: str, width=256, height=256):
    """Generate one fractal, retrying until ≥90% of pixels have value >10."""
    while True:
        center = sample_center()
        zoom     = random.uniform(*ZOOM_RANGE)
        max_iter = random.randint(*ITER_RANGE)

        re_start = center.real - zoom
        re_end   = center.real + zoom
        im_start = center.imag - zoom
        im_end   = center.imag + zoom

        img = np.zeros((height, width), dtype=np.uint8)

        for x in range(width):
            for y in range(height):
                zx = re_start + (x/(width-1))*(re_end-re_start)
                zy = im_start + (y/(height-1))*(im_end-im_start)
                z = 0+0j
                it = 0
                while (z.real*z.real + z.imag*z.imag) <= 4 and it < max_iter:
                    z = z*z + complex(zx, zy)
                    it += 1

                if it < max_iter:
                    log_zn = np.log(z.real*z.real + z.imag*z.imag) / 2
                    nu     = np.log(log_zn/np.log(2)) / np.log(2)
                    val    = it + 1 - nu
                    color  = int(255 * val / max_iter)
                else:
                    color = 0

                img[y, x] = color

        # check if ≥90% of pixels have value >10
        if np.count_nonzero(img > 10) >= DETAIL_THRESHOLD * width * height:
            Image.fromarray(img).save(
                os.path.join(output_dir, f'fractal_{idx:05d}.png'))
            break
        # else: retry with new parameters

def main(num_images=30000, output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    worker = partial(generate_and_save, output_dir=output_dir)
    with Pool(cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(worker, range(num_images)),
                      total=num_images, desc="Generating fractals"):
            pass

if __name__ == '__main__':
    main()

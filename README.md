# GPGPU

**Authors**:
- Jérôme Dubois
- Lukas Rabier
- Laurélie Michielon
- Théotime Terrien

Data should be placed in *data/*, which is ignored by git.

# Build

* `mkdir build && cd build && cmake .. && make -j8`

# Benchmarks

* Usage `./run $data`. data can be a image or a folder with multiple images.
  The benchmarks will be individually run for each image. The results are store
  in a csv (one by image) in a `results-\*` folder.

* Then you can use `visu_benchmark.ipynb` to visualize them.

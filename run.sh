#!/bin/sh

[ $# != 1 ] && echo 'Missing args: expect ./run.sh $data_path' && exit 0

data_path="$1"

progress_bar() {
  local width=20
  to_draw=$(($1 * width / $2))
  space=$((width - to_draw))

  printf '['
  printf '#%.s' $(eval "echo {0.."$(($to_draw))"}")

  if [ "$space" != "0" ]; then
    printf ' %.s' $(eval "echo {0.."$(($space))"}")
  fi

  printf "] $(($1 * 100 / $2))%%\r"
}

# Make de different results directory each time the scrip is run.
number=0
result_dir='results-00'
while [ -e "$result_dir" ]; do
    printf -v result_dir '%s-%02d' results "$(( ++number ))"
done

mkdir "$result_dir"

# Get CPU and GPU info.
cat /proc/cpuinfo >> "$result_dir"/log.txt
nvidia-smi > "$result_dir"/log.txt

# Find all jpg in data
images=$(find "$data_path" -type f -name \*.jpg)
nb_img=$(echo "$images" | wc -l)
cur=0

# Execute the benchmark on all images.
for img_path in $images
do
  progress_bar cur nb_img
  f=$(basename -- "$img_path")
  result="${f%.*}"

  BENCH="$img_path" ./bench --benchmark_format=csv \
    2>> "$result_dir"/log.txt \
    1> "$result_dir/""$result".csv

  cur=$((cur + 1))
done

progress_bar cur nb_img
# Do not rewrite on the progress bar
echo ""

# Do the same for videos ???

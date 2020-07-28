# Kernel-Based-Neural-Ranking-Models

### Output

This folder stores the outputs of the inference phase.

When running the inference code, two files will be generated:

- `*_raw_output.txt`: this file shows the relevant score on each infered (query, doc) pairs.
- `*_output.txt`: this file shows the ranking result on relevant documents for each query. The format is corresponding to the MSMARCO official evaluation script.

### Format

```shell
# _raw_output.txt
qid \t did \t score

#_output.txt
qid \t did \t rank
```





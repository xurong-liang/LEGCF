# Code repo for Lightweight Embeddings for Graph Collaborative Filtering (LEGCF)

- The paper can be found [here](https://arxiv.org/abs/2403.18479).
- Accepted by SIGIR'24.
- Code structure adopted from [LightGCN](https://github.com/gusye1234/LightGCN-PyTorch).

## Dataset
- Train and test sets of Gowalla, Yelp2020 and Amazon-book are located in [here](./data).

---

## Quick Start

Run program with Gowalla dataset with bucket size 500, 2 compositional meta-embeddings/entity on cuda device 0:
```shell
python3 engine.py --dataset_name gowalla --num_clusters 500 --num_composition_centroid 2 --device_id 0
```


---

## Implementation Details

1. Gowalla

   | Hyperparam               | Value       |
   |--------------------------|-------------|
   | GCN layer                | 3           |
   | l2 penalty factor        | 5           |
   | lr                       | 1e-3        |
   | assignment update freq $m$  | every epoch |
   | \#composition embs per entity $t$ | 2  |
   | \#clusters         $c$       | 500     | 
   | init anchor weight $w^*$  | 0.5     |
   
2.  Yelp2020 

    | Hyperparam                   | Value       |
    |------------------------------|-------------|
    | GCN layer                    | 4           |
    | l2 penalty factor            | 5           |
    | lr                           | 1e-3        |
    | assignment update freq  $m$  | every epoch |
    | #composition embs per entity $t$ | 2       |
    | #clusters            $c$     | 500         |
    | init anchor weight $w^*$  | 0.6          |

3. Amazon-book

   | Hyperparam                   | Value       |
    |------------------------------|-------------|
    | GCN layer                    | 4           |
    | l2 penalty factor            | 5           |
    | lr                           | 1e-3        |
    | assignment update freq  $m$  | every epoch |
    | #composition embs per entity $t$ | 2       |
    | #clusters            $c$     | 500         |
    | init anchor weight $w^*$  | 0.9          |


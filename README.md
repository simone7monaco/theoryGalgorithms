# Theory-guided deep learning algorithms: an experimental evaluation
by Simone Monaco, Daniele Apiletti and Giovanni Malnati

## Abstract
The use of theory-based knowledge in machine learning models has had a major impact on many engineering and physics problems. The growth of deep learning algorithms is closely related to an increasing demand for data that is not acceptable or available in many use cases. In this context, the incorporation of physical knowledge or a-priori constraints has proven beneficial in many tasks. On the other hand, this collection of approaches is context-specific, and it is difficult to generalize them to new problems. In this paper, we experimentally compare some of the most commonly used theory injection strategies to perform a systematic analysis of their advantages. Selected state-of-the-art algorithms were reproduced for different use cases to evaluate their effectiveness with smaller training data and to discuss how the underlined strategies can fit into new application contexts.

## Reproducing the results
A python script allow to run each of the experiments as follow.
```
python main.py <exp> [OPTION]
```

Where `exp` stays for:
- **Lake Temperature**, the code is an extension of [[1]](https://github.com/arkadaw9/PGNN) and [[2]](https://github.com/arkadaw9/PGA_LSTM).
  - With physical loss function: `pgnn`
  - Without physical loss function: `pgnn0`
  - With PGA-LSTM: `pga`
- **Convective movements in climate modeling**, extended from [[3]](https://github.com/raspstephan/CBRAIN-CAM). The full dataset has been requested to original authors.
  - Basic MLP `cbrain1`
  - Hard constraints `cbrain2`
  - Soft constraints `cbrain3`
- **Climate prediction**, extended from [[4]](https://github.com/sungyongs/dpgn)
  - With physical loss function: `dpgn`
  - Without physical loss function: `dpgn0`
Moreover, experiments can have extra arguments as options (run `python main.py <exp> --help` to see all of them).

## License
All source code is made available under a BSD 3-clause license. You can freely use and modify the code, without warranty, so long as you provide attribution to the authors.

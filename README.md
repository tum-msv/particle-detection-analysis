# Joint Particle Detection and Analysis by a CNN and Adaptive Norm Minimization Approach

This is the simulation code to the article

M. Baur, M. Reisbeck, O. Hayden, and W. Utschick, _Joint Particle Detection and Analysis by a CNN and Adaptive Norm Minimization Approach_, Submitted for possible publication.

## Abstract
Optical flow cytometry is used as the gold standard in single cell function diagnostics with the drawback of involving high complexity and operator costs. Magnetic flow cytometers try to overcome this problem by replacing optical labeling with magnetic nanoparticles to assign each cell a magnetic fingerprint. This allows operators to obtain rich cell information about an unprocessed biological sample at near in-vivo conditions in a decentralized environment. A central task in flow cytometry is the determination of cell concentrations and cell parameters, e.g. hydrodynamic diameter. For the acquisition of this information, signal processing is an essential component. Previous approaches mainly focus on the processing of one-cell signals, leaving out superimposed signals originating from cells passing the magnetic sensors in close proximity. In this work, we present a framework for joint cell/particle detection and analysis, which is capable of processing one-cell as well as multi-cell signals. We employ deep learning and compressive sensing in this approach, which involves the minimization of an adaptive norm. We evaluate our method on simulated and experimental signals, the latter being obtained with polymer microparticles. Our results show that the framework is capable of counting cells with a relative error smaller than 2 %. Inference of cell parameters works reliably at both low and high noise levels.

## File Organization
Our experimental data can be found under `python/data/exp_data_4um8um.mat`, together with the extracted cell parameters from this data. A few model checkpoints, including our best performing FCN-2 from the results section, are located at `python/models`. Simulated training data for a FCN may be generated with the file `createDataset.m` in the `matlab` directory.

## Implementaion Notes
This code is written in _Python_ and _Matlab_. It uses the deep learning library _PyTorch_.
The code was tested with _Python_ version 3.7, _PyTorch_ version 1.5.1, _Matlab_ version 2019b, on a _Ubuntu_ 18.04.5 system.

## License
This code is licensed under 3-clause BSD License:

>Copyright (c) 2021 M. Baur.
>
>Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
>
>1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
>
>2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
>
>3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
>
>THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

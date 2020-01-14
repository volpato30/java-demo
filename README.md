# PyTorch Java Demo

This repository is a demonstration of how to use PyTorch from Java.

## Setup

Download and unpack libtorch nightly (or 1.4 or greater once it is released).
From the pytorch.org homepage under "Quick Start Locally",
make sure "LibTorch" is the selected package.
As of this writing, only Linux is supported.  Mac and Windows will follow.


Change `java.library.path` in `build.gradle` to `/path/to/libtorch/lib`.

Run `./gradlew run` to build and run the demo application.
It will load `demo-model.pt` and run it on some simple data.

More information about the Java API and TorchScript:
- [TorchScript tutorial](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)
- [TorchScript reference](https://pytorch.org/docs/stable/jit.html)
- [Loading TorchScript in C++](https://pytorch.org/tutorials/advanced/cpp_export.html)
- [PyTorch-style JavaDoc](https://pytorch.org/docs/stable/packages.html)
- [Standard JavaDoc](https://dreiss.github.io/pytorch/android/pytorch_android/javadoc-out/)
- [PyTorch Android tutorial](https://pytorch.org/mobile/android/)

# Data-Parallel Types

SSE4.2 implementation of [chapter 9 Data-Parallel Types](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/n4808.pdf)

# Code Coverage

```
llvm-profdata-8 merge -sparse default.profraw -o default.profdata
llvm-cov-8 report -ignore-filename-regex='usr/src/googletest/.*' ./unit_tests -instr-profile=default.profdata
llvm-cov-8 show -format=html --output-dir=cov/ -ignore-filename-regex='usr/src/googletest/.*' ./unit_tests -instr-profile=default.profdata
```

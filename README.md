# CXYNN

# debug
1. 不能在main函数之前调用cuda的运行时库。通过在构造函数中调用cuda的运行时库并申明全局变量即可做到这一点。
2. 使用cudaMemcpy函时，只要有一处cudaMemcpy函数的方向写反了，就会造成在调试的时候所有的cudaMemcpy函数都不可用.

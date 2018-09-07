# CXYNN

# debug
1. 不能在main函数之前调用cuda的运行时库。通过在构造函数中调用cuda的运行时库并申明全局变量即可做到这一点。

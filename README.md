# GPU_Pollard-Rho
一个很基础的研究项目，利用GPU跑Pollard's Rho算法来分解大质因数     
## 对于源码  
直接运行以下命令即可编译  
`nvcc -arch=sm_89 -I x64-windows/include -I cgbn -L x64-windows/lib GPUPollard_Rho.cu -lgmp -o GPUPollard_Rho -allow-unsupported-compiler`   
其中的`-arch=sm_89`是专门为RTX4060优化的，对于更老的显卡可能不支持那么高的版本，适当修改即可  
## 对于程序
将*GPUPollard_Rho.exe*与*gmp-10.dll*、*gmpxx-4.dll*文件放在同一文件夹下即可  
具体使用,打开CMD输入以下指令即可分解  
`GPUPollard_Rho.exe 666666666666666`   
注意：这是针对4060编译出来的windows平台程序，不能保证通用性 
### 不建议输入超过40位的数（相应为128bit多），这个算法的效率会急速下降以至于根本算不出来
### （最后感谢伟大的Gemini3-Pro助力）

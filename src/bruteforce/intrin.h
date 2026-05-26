#pragma once

#include <cpuid.h>
#include <x86intrin.h>

#ifdef __cpuid
#undef __cpuid
#endif

inline void __cpuid(int cpu_info[4], int function_id)
{
	unsigned int eax = 0;
	unsigned int ebx = 0;
	unsigned int ecx = 0;
	unsigned int edx = 0;
	__cpuid_count(function_id, 0, eax, ebx, ecx, edx);
	cpu_info[0] = static_cast<int>(eax);
	cpu_info[1] = static_cast<int>(ebx);
	cpu_info[2] = static_cast<int>(ecx);
	cpu_info[3] = static_cast<int>(edx);
}

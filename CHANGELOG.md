# VSCMG DRL 控制系统迭代日志

## [v0.1.0] - 物理引擎与动力学底座竣工
**日期: 2026-04-09**

### Added (新增)
- **底层物理接口 (`geometry/base_config.py`)**: 实现了 VSCMG 阵列的抽象基类 `BaseVSCMGArray`，确立了控制力矩的底层计算法则。
- **金字塔构型 (`geometry/pyramid_config.py`)**: 实现了标准的四飞轮金字塔构型，精确推导了自旋轴与横向轴的雅可比矩阵，并通过了静态零点测试。
- **航天器动力学 (`envs/dynamics.py`)**: 基于牛顿-欧拉方程，实现了刚体姿态动力学求解器，完美解耦了 VSCMG 的牵连耦合力矩。

---

## [v0.0.0] - 工程地基初始化
**日期: 2026-04-09**

### Added (新增)
- 建立 7 大核心文件夹骨架 (`agents`, `configs`, `envs`, `geometry`, `classical_control`, `utils`, `checkpoints`)。
- 部署 `.gitignore` 防御结界，排除运行缓存、IDE 配置和大型权重文件。
- 确立面向对象的构型基类思想。
- 绑定 GitHub 远程仓库并完成首次纯净工程骨架的推送与 v0.0.0 里程碑打标。

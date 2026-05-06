## Execution Safety Rules

1. 当用户或指令明确写着"不生成文件""不创建临时脚本""只允许 inline Python""只读诊断"时，严禁使用：
   - `cat > /tmp/*.py`
   - `cat > _tmp_*.py`
   - `python script.py`
   - 任何会落地 `.py`、`.txt`、`.npz`、图像或日志文件的替代方案

2. 如果 heredoc 因引号或 delimiter 问题失败，不允许自动改成写临时文件。应优先改用不带引号的 heredoc，例如：

   ```bash
   python <<XEOF
   # Python code here
   XEOF
   ```

3. 如果 inline Python 命令仍然因为 shell quoting 失败，应停止并汇报，不要自行创建文件绕过限制。

4. 对于本项目的诊断类任务，默认遵守：

   * 不训练
   * 不运行 `train.py`
   * 不修改代码，除非本轮明确允许
   * 不生成文件，除非本轮明确允许
   * 不 commit / push / tag / release，除非本轮明确允许
   * 不改 reward
   * 不改 TD3 update
   * 不改环境源码
   * 不改 PyramidVSCMG 几何公式

5. 如果本轮任务明确允许修改某个文件，只能修改该文件和该任务指定的范围；不要扩大修改范围。

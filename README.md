# pairwise-loudness

音频中间件做响度归一化时，不同关卡音量忽大忽小：LUFS 积分窗口没有按 gating 阈值排除静音段，安静场景被拉得很响；true peak 只看了采样点，重建峰值超限，母带削波；限幅器 release 固定，鼓点密集时泵浦；5.1 和立体声共用一套增益，中置人声被压扁；采样率转换后响度漂移，没有重新积分；ReplayGain 标签写的是旧版本，播放器读不到。audio/loudness.py 是核心，tests/test_loudness.py 15个红。要修的是：按 ITU-R BS.1770 做 gated LUFS；true peak 过采样检测；限幅器自适应 release；按声道布局分别归一；重采样后重算；写标准 ReplayGain 标签。

## 运行

```bash
python -m pytest tests -q
```

初始状态：15 个用例失败。修复实现使全部用例通过，不要修改 `tests/` 下的测试。

语言/框架：Python 3, pytest；环境可复现等级：无外部依赖。

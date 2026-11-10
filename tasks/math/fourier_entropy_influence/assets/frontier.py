"""The best known construction for this task: Numaro's balanced logic-monotone function on 18 variables (zenodo
21497769, teorth/optimizationproblems constant 71a [Num2026], July 2026), from Mosaic Intelligence's n = 17 function
(zenodo 21113935) by splitting the one four-cell auxiliary variable 2 + 2 with a new 18th variable. Exact influence
I = 261/128 and H / (I - 1) = 6.5218457109230465756..., a strict local optimum over the 16,838 single cell moves of
its authors' class. DATA is its truth table (the archive's true_hex, which is assets/fei.py's hex convention), zlib-compressed and
base64-encoded."""

import base64
import zlib

DATA = (
    "eNrFnWtqHDgQBq9UP3OdYZi5/xHWWRtinN15SKqumGBMYopm1BSt5kP32/3bF9xvl8ufv79/vt+uH38+v3/++/X669fn989///eL31/ffp/L189f"
    "v//x08d//+vPfZBPwb/er2n93/j/X/9tk3/b/Pxn+En9L33+M/yR83/5+L1L2P8P+En90/3/gD9y/pc+/xl+Uv90/y99/oZ/iP1P7H9i/xP7n9j/"
    "xP4n9j+x/4n9T+x/Yv8T+5/Y/8T+J/b/2/1/ecK/yPO/zk/qf73/df6T87/BPzL/6/yk/tf7X+ePnP/1+V/nJ/VP9//6/H/m/K3P/zo/qX+6/9fn"
    "/8P+IfY/sf+J/U/sf2L/E/uf2P/E/if2P7H/if1P7H9i/xP7n9j/zv3f7Yt/i/b/O/yk/sP3fzv8E/df/82f2//v8JP6D9//7fBHzr+8/9/hJ/VP"
    "97+8/396/uT9/w4/qX+6/+X9/+v+IfY/sf+J/U/sf2L/E/uf2P/E/if2P7H/if1P7H9i/xP7n9j/zv2fOP/r/KT+w/d/4vy/wZ/b/4vzv8M/fP8n"
    "zv9nzp+8/xfnf4c/3f/y/l+c/3V+Uv90/8v7f3H+1/lJ/Zn/if1P7H9i/xP7n9j/xP4n9j+x/4n9T+x/Yv8T+9/M/3z/XuT/1/hJ/Ur+Z41/Lv/y"
    "kz+d/1/jJ/Ur+Z81/sj5H8n/r/GT+qf7fyT//+D8jeT/1/hJ/dP9P5L/f8U/xP4n9j+x/4n9T+x/Yv8T+5/Y/8T+J/Y/sf+J/U/sf2L/E/vfzP8o"
    "87/OT+pX8j/K/L/Bn87/K/O/w1fyP8r8f+b8jeT/lfnf4U/3/0j+X5n/dX5S/3T/j+T/lflf5yf1Z/4n9j+x/4n9T+x/Yv8T+5/Y/8T+J/Y/sf+J"
    "/U/sfzP/8+d7k/9f4yf1K/mfNf65/MtP/nT+f42f1K/kf9b4I+d/JP+/xk/qn+7/kfz/g/M3kv9f4yf1T/f/SP7/Ff8Q+5/Y/8T+J/Y/sf+J/U/s"
    "f2L/E/uf2P/E/if2P7H/if1P7H8z/6PM/zo/qV/J/yjz/wZ/Ov+vzP8OX8n/KPP/mfM3kv9X5n+HP93/I/l/Zf7X+Un90/0/kv9X5n+dn9Sf+Z/Y"
    "/8T+J/Y/sf+J/U/sf2L/E/uf2P/E/if2P7H/F+7/rj/41wPvz79z/yfzk/rfuf+T+U/vv5b5h/b/Mj+p/537P5k/cv539v8yP6l/uv939v8nzt/O"
    "/l/mJ/VP9//O/v+of4j9T+x/Yv8T+5/Y/8T+J/Y/sf+J/U/sf2L/E/uf2P/E/if2/8L932P+RZ7/dX5S/zv3fzL/6f3XMv/Q/l/mJ/W/c/8n80fO"
    "/87+X+Yn9U/3/87+/8T529n/y/yk/un+39n/H/UPsf+J/U/sf2L/E/uf2P/E/if2P7H/if1P7H9i/xP7n9j/xP537v/E9/90flL/4fs/8f2/Df7c"
    "/l98/8/hH77/E9//O3P+5P2/+P6fw5/uf3n/L77/p/OT+qf7X97/i+//6fyk/sz/xP4n9j+x/4n9T+x/Yv8T+5/Y/8T+J/Y/sf+J/e/c/4nzv85P"
    "6j98/yfO/xv8uf2/OP87/MP3f+L8f+b8yft/cf53+NP9L+//xflf5yf1T/e/vP8X53+dn9Sf+Z/Y/8T+J/Y/sf+J/U/sf2L/E/uf2P/E/if2P7H/"
    "zfyP8v6fzk/qV/I/yvt/G/zp/L/y/p/DV/I/yvt/Z87fSP5fef/P4U/3/0j+X3n/T+cn9U/3/0j+X3n/T+cn9Wf+J/Y/sf+J/U/sf2L/E/uf2P/E"
    "/if2P7H/if1P7H8z/6PM/zo/qV/J/yjz/wZ/Ov+vzP8OX8n/KPP/mfM3kv9X5n+HP93/I/l/Zf7X+Un90/0/kv9X5n+dn9Sf+Z/Y/8T+J/Y/sf+J"
    "/U/sf2L/E/uf2P/E/if2P7H/zfyP8v6fzk/qV/I/yvt/G/zp/L/y/p/DV/I/yvt/Z87fSP5fef/P4U/3/0j+X3n/T+cn9U/3/0j+X3n/T+cn9Wf+"
    "J/Y/sf+J/U/sf2L/E/uf2P/E/if2P7H/if1P7H8z/6PM/zo/qV/J/yjz/wZ/Ov+vzP8OX8n/KPP/mfM3kv9X5n+HP93/I/l/Zf7X+Un90/0/kv9X"
    "5n+dn9Sf+Z/Y/8T+J/Y/sf+J/U/sf2L/E/uf2P/E/if2P7H/X/v8/wFhuSbe"
)


def candidate() -> dict:
    return {"n": 18, "table": zlib.decompress(base64.b64decode("".join(DATA))).decode()}

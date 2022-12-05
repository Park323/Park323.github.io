---
title: "[Python] 코드 안에서 command line을 실행하고 싶을 땐?"
date: 2022-12-05 18:20:00 -0400
categories: Implementation
---

## Issue

```python main.py -c ...```
같은 command를 반복해서 실행해야 하는데,, python의 for loop에서 한 번에 실행시키고자 한다.

## Solution

시도

Help me Google~!~

최종

python subprocess 사용하면 된다.

https://docs.python.org/ko/3/library/subprocess.html

---
title: "[Python] Numpy의 recarray"
date: 2022-12-20 14:40:00 -0400
categories: Implementation
---

## Issue

다른 repo에서 preprocessing된 feature를 저장할때 사용하는 자료형을 살펴보니, numpy.recarray라는 처음 보는 타입임을 발견.

## Solution

recarray는 속성을 사용해서 field 접근을 가능하게 하는 자료형이다.

몰랐었는데, 그냥 ndarray 선언할 때도 dtype에 tuple을 활용하면 field를 설정해줄 수 있다. ex) dtype=[('x', '<i8'), ('y', float)]

python에서 zip으로 쌍을 만드는 것과 비슷해보이고, field name을 정해주기까지 할 수 있어서 좋아보인다.

결국 recarray의 핵심은 그저 array의 field에 attribute으로 접근이 가능하다는 것인듯. ex) array_1.x

발견한 repo에서는 (filename, audio, caption)의 쌍을 npy로 저장했는데, 데이터셋을 다룰 때 쓸 수 있을 듯 하다.

https://numpy.org/doc/stable/reference/generated/numpy.recarray.html


# 1. Introduction

Image coregistration : 2개 이상의 image volume을 같은 기하 공간에 정렬

* intramodality : same image modaltiy
* intermodality : across modality
* intersubject : various subject to a common space


clinical purpose로 사용

* rigid body
    * 영상 사이에 모양에 변화나 변형이 없다. 
    * neuroimaging에서 intrasubject coregistration이 급부상
    * 가장 잘 검증되어있다.
    * 6개의 free degree
        * x, y, z 따라 translate
        * x, y, z 라 rotation
* non rigid body
    * 특히 뒤틀림에 대한 비선형 모델 기반
    * 계산적으로 복잡하고 검증 어려움


이미지의 뒤틀림이나 변형의 방법은 clinical coregistration에서 점점 더 사용될 것으로 기대

dementia, 다발성 경화증 (MS), 중추신경계 (CNS) 종양, 뇌전증등의 neuroimaging에 clinical application


# 2. Intramodality Coregistration

serial imaging - obvious clinical application

다양한 시간 간격을 두고 얻은 환자의 스캔을 비교하여 병의 진행과 치료의 반응을 추적.

방사선과에서는 imaging scan을 일상적으로 register하지 않는다.

일치하지 않는 film montage를 보고 질병의 변화 평가위해 최선을 다한다.

총제적 변화의 경우, 최적은 아니지만 적절할 수 있다.

그러나 미묘한 변화의 경우, 맞지 않는 이미지에 대한 visual comparison은 적절하지 않을 수 있다.

이러한 이유 때문에 registration을 잘 적용하지 않는다.

이와는 대조적으로, 유효한 방법이 개발되자마자, 신경 영상 연구자는 이미지 변수의 객관적이고 측정가능한 변화를 연구하기 위해 serial image에서의 coregistration기술을 재빨리 이용했다.


* Goodkin and colleagues 

MRI에서 재발 이장성 다발성 경화증 환자에 대해 새로운 contrast enahancement lesion이 발생하거나, 발생하지 않거나에 대한 뇌백질 신호 변화의 양적인 연구 

Fig 1. 

![image](https://user-images.githubusercontent.com/101063108/163732572-8ff4b44c-f63a-47b4-8b04-37b48c5a1a4e.png)


dynamic nature of contrast enhancing lesion in MS (6 month)

단순히 연속적인 질병 변화를 영상의 정확한 정렬과 비교하는 명백한 가치를 보여줌


serial imaging에서 coregistration을 더 잘 활용하기 위해 subtraction techniques를 사용한다.

한 regist된 이미지에서 다른 이미지로 subtraction하는 것은 serial image에서 변경사항을 다른 수준으로 시각화하는 다른 이미지를 만든다.

다른 방법으로는 알 수 없는 변화도 종종 발생한다.

Image subtraction : 이미지 복셀 단위로 신호 강도를 단순하게 산술적으로 감산한다.

하나의 스캔에서 다른 것까지의 global values change는 dataset에 대한 intensity value의 정규화가 필수이다.


이것은 individual intensity값을 전체 dataset의 평균에 참조함으로써 해낼 수 있다.

subtraction에 의한 미묘한 변화를 정의, 동시에 약간 잘못 정렬되어 있는 것은 제외하기 위해, 특히 뇌표면 근처 영역에서 매우 정확한 subvoxel registration이 필요하다.

Fig 2

![image](https://user-images.githubusercontent.com/101063108/163732740-541481d6-63f2-4a7a-9c1b-cf411ed5804e.png)

왼쪽 측두엽 뇌종양의 외과적 절제 후 환자의 serial MRIs registration

regist에 의한 영상 정렬은 IV(Intravenous) contrast enhancement의 작은 focal region을 식별 할 수 있게 한다.

regist전에, 슬라이스 별 정확한 시각적 비교가 불가능했기 때문에, 한 스캔에 대해서 개선사항이 있고, 다른 스캔에 대해 개선사항이 없다고 말할 수 없다.

* Hajinal et al 

고해상도 뇌영상에서 변하지 않는 구주물로부터의 신호를 완전히 취소하는 subtraction와 subvoxel registration

연속된 영상들 ( 정상적인 뇌 성숙과 발달, 종양크기의 작은 변화, 작은 두부외상과 관련된 뇌의 작은 이동)에 대한 미묘한 변화를 감지

* Curati et al

IV contrast enhancement와 관련된 변화를 탐지하는 증가된 민감도를 보여줌

coregister subtraction에서 CNS 염증성 질환, MS 병변, 정맥 및 악성 종양의 증강정도와 분포에 관한 추가적이고 분명하지만 미묘한 변화를 탐지할 수 있다.

coregistration & subtraction은 특히 다음에서 가치가 있다 : 

(1) 작은 정도의 enhancement의 인지

(2) 매우 낮거나 높은 baseline 신호를 가진 조직이나 유체의 enhancement

(3) 복잡한 해부학적 인터페이스, 경계 다른 영역에서의 enhancement

(4) 얇은 slice가 얻어졌을 때의 enhancement의 평가


또한 뇌출혈과 조직혈류 분포 각각의 연속성 평가에서의 민감도 및 확산 가중 시퀀스가 있는 serial image에 coregistration과 subtraction을 적용하는 잠재적 가치에 대해서도 주목할만 하다.


SPECT를 사용한 focal 뇌전증(뇌의 제한된 부분에서 시작하는 발작을 수반하는 뇌전증)의 국소화

- resgist &  subtraction 적용을 위해 최근에 인정된 분야


여러종류(HMPAO, ECD)의 방사선 추적기를 사용하여 부분발작 중 국소적인 뇌 혈류 변화를 시각화 할 수 있다.

HMPAO/ECD는 정맥 주사 후 30-60초 이내에 최대흡수를 달성한 1차 통과 뇌추출율을 가진다.

이때 동위원소가 뇌에 갇혀, 발작동안 뇌 혈류량 사진을 찍을 수 있다.

비교적 긴 반감기로, 최대 3-4시간 후 편리하게 이미지를 생성한다.

coregistration & subtraction 의 ictal (발작 동안) SPECT imaging에서의 한계

1) 상대적으로 낮은 공간 분해능 -> CT나 MRI의 coregistration 없이는 해부학적 디테일을 해석할 수 없다.
2) 기준치의 발작 사이의 스캔에서 개개인의 차이가 어려운 비교를 만든다. - 특히 발작 시작영역에서 뇌혈류에서의 상대적 focal 감소의 발작 사이 영역을 가진다.
3) 발작 중과 발작 사이에서의 복잡한 visual 비교는 신호강도와 일치하지 않는 슬라이스 방향의 전역적 차이

Fig 3.

![image](https://user-images.githubusercontent.com/101063108/163733440-cc636cdb-3cfa-402b-9b9c-3509c0af2a9f.png)

위의 문제점들, 그리고 그에 대한 수정 ( coregistration, normalization, subtraction이용)

* O'Breien and colleagues

subtraction ictal SPECT coregisted to MRI (SISCOM) 뇌전증 중심으로 널리 사용

* Holmes et al 

신호 평균을 통한 이미지 향상

신경해부학적 세부사항을 더 잘 볼 수 있게 하는 향상된 대비와 더 나은 공간해상도는 signal averaging을 사용하여 MRI에 적용하여 얻어질 수 있다.

신호의 증가는 기여 스캔 수의 루트에 따라 증가할 것

Fig 4.

![image](https://user-images.githubusercontent.com/101063108/163734134-e9c1def1-dbb0-4bf1-ac44-e883362f3fcc.png)

뇌전증 환자의  평균 4개의 MRI 스캔으로부터의 signal to noise에서의 두배의 개선을 보여준다.

백색질에서의 잡음의 상대적 감소로 회색질 기반 병변을 더 잘 정의할 수 있어졌다.

임상신경영상의 신호평균화에 대한 registration은 완전히 잘 이용된다.

이 응용은 공간분해능을 개선할 수 있는 변수를 포함하여, 최적화 할 수 있는 수많은 변수를 가진 새로운 응용을 가진다.

EX1. 감소된 복셀 크기로신호 손실을 보상하기 위해 신호 평균 사용

EX2. 가만히 있을 수 없는 환자의 이미지화하는데 사용되는 빠른 획득된 스캔에서의 신호 감소 보상

신호 평균을 잘 활용할 수 있는 최적의 시퀀스는 여전히 개발이 필요하다.

# Intermodality ( or Multimodality) Coregistration

functional & structure imaging - 각 독립적인 분석으로부터 구할 수 없는 고유한 정보를 제공

MRI 또는 CT영상에서의 regist된 고해상도 해부학은 functional image data 해석을 위한 훨씬 더 정확한 해부학적 기초를 제공.

-> regist된 functional 영상은 모호하거나 비특이적인 구조적 병변 및 MRI/CT의 기타 이상증상의 임상적 중요성을 해석하는데 도움을 줄 것.

Intermodality coregistration이 매우 중요한 치료 결정에 대한 임상 영상의 해석을 도울 수 있다.

* Nelson et al

뇌종양 환자 평가에서의 coregist하는 volumetric MRI & 고상도 FDG-PET의 가치를 보여줌

MRI contrast enhancement 변화에 의존 - 뇌손상으로 인한 뇌종양-재발적 변화 때문에 오해의 소지가 있을 수 있다.

활성 종양 : MRI에서는 잘 보이지는 않지만, FDG-PET에서는 대사적으로 활성화됨.

일반 회질보다 낮거나 같은 FDG 흡수수준에서 이상상태와 정상적인 해부학을 왜곡하는 피질에 가까운 contrast enhancing lesion

Intermodality image coregistration은 방사선 치료 계획에도 유용

* Rosenman et al

방사선 치료의 중요 단계 시뮬레이션 film으로 종양 체적을 정확히 regist한다. 

그렇지 않을 경우 방사선 표적이 정상 범위를 포함하거나 종양을 놓치게 된다.'


coregistration의 목표는 이전에 획득한 비계획적 study(고해상도 사전 개입 이미지)에서 계획적 X선 CT 스캔으로 데이터를 전달하는 것이다.

planning CT에 전체 종양이 표시가 안될 수 있다. 원래 종양부피 대상으로 사전 스캔이 필요하다.

종양조직 단백질에 결합하는 단일 복제 항체를 사용하여 생성된 SPECT스캔에 적용이 가능하다.

FDG-PET에서 미묘한 이상을 감지하기 위한 임계값을 낮춘다면,, 해부학적 비대칭성 교정,  구조적 이상의 결함 원인 여부 확인, 의심부위가 부분 부피 평균에 의해 영향을 받는지 판단하기 위해, MRI coregistration는 필수적이다.

Fig 5.

![image](https://user-images.githubusercontent.com/101063108/163734953-75126b8c-12ce-46f2-a467-1aa316aaeb92.png)

coregistration MRI 해부학과 부분 부피 평균 보정을 통해서만 왼쪽 해마에서 분명한 focal 대사 저하 확인


* EEG / MEG 국소화 :  간질 발생 영역 식별 위해 functional, structural  해부학적 영상과 결합 가능하다.

MRI 에서 간질성 병변을 식별할 수 없을 때 유용하다.

MRI 영상에서 병리적인 것이 보이지 않고, 모호하면 functional imaging과 전기 생리학적 국소화의 multimodality coregisration을 통해 간질 유발 영역의 식별정확도를 높일 수 있다.

Fig 6.

![image](https://user-images.githubusercontent.com/101063108/163735012-f6a9d070-34c0-4f1e-8edf-d2cb88e2040f.png)

multimodality coregistration에서만 감지된 원인 불명의 병리 예시

EEG에 기록된 임상적 특징 : 빈번한 간질성 방전(스파이크)에 기초하여 우측 측두엽에서 발생한 것으로 의심되는 발작.

MRI에서의 병변 이상은 없었고, FDG-PET에서도 정상 스캔으로 보여짐.

전극 기록 : 발작발생의 반구의 국소화 x, 정의 x

MEG : 우측 측두엽 focal region에 간질형 이상 3D위치 추정

coregistrated된 FDG-PET를 재검사 : MEG 추정치와 국소적 상대 초점 대사저하를 분명히 보여준다.


fMRI 또는 PET 기반의 뇌 매핑과 추가로 결합하면, 완전히 비침습적 수술 전 뇌전증 평가를 구성할 수 있다.

Multimodal coregistration : PET, fMRI, EEG, MEG로 뇌 매핑 연구를 수행하는 실험실에서 널리 사용된다.

상대적으로 높은 공간분해는 image modality(fMRI, PET)의 functional localization에 대한 사전지식을 높은 시간 분해능 modality 의 source localization의 활용에 도움이 되어야 한다.

신경 병리학적 위치 확인 작업의 수행을 위해 전극 영상 기록이 필요하다.

EEG를 사용하면 두피와 두개내 전극 위치를 MRI에 regist할 수 있다.

두피표면 : 지표면 등고선, landmark, 기준 marker가 coregistration에 사용됨

두개 내 : 단순히 전극을 이식한 환자를 영상화하여 다른 영상 양식에 regist가능

전극의 영상화 : MRI . CT . 두개골 방사선 사진

MRI ->  비강자성 니켈크롬 합금 특수 전극사용 (비쌈) : 전극이 이동할 위험을 제거

CT/MRI : artifact는 접점의 정확한 시각화를 보기 어렵게 한다.

artifact 제거하고 비용을 절감하는 방법은 이식전 MRI에 이식 후 CT를 coregistration하는 것이다. (Fig 7.)

![image](https://user-images.githubusercontent.com/101063108/163735421-f7e78227-6bbf-4423-b91a-b46ca031d569.png)


coregistration 영역은 neuroimaging 기술에서의 급속한 발전으로 점점 더 어려워지는 진단 방사선의 한 측면인 영상신호 변화의 기본 병리를 정확히 이해하는데 큰 가능성을 지닌다.

# Intersubject Coregistration

intersubject image registration은 통제 집단의 평균 이미지를 생성

관심 환자의 모집단 내 또는 그것에 걸친 병리학적 영상 이상을 비교가능

SPECT , PET 영상 - 가장 일반적인 임상응용

* Richardson and Koepp et al

두개의 부분 뇌전증 환자 그룹에서, flumazenil PET와의 central benzodiazepine binding 이상을 조사.

statistical parametric mapping을 사용, 각 환자의 정상 및 비정상 benzodiazepine binding 분포를 객관적으로 측정

coregisted MRI는 분할된 MRI의 convolving한 회질을 사용하여 부분 volume 효과에 대한 중요한 보정을 하게한다.


* Bartenstein et al

치매에서 SPECT 영역 뇌 혈류 장애 평가는 intersubject coregistration에 의해 허용되는 해부학적 표준화의 이점을 보여준다.

-> 알츠하이머 병 환자의 CBF 변화의 패턴과 심각도에 대한 관찰자-독립적 평가를 더 정확히 한다.

functional imaging의 진단 가치에 큰 어려움이 있는 조기 또는 경증 치매에서의 유의한 이상 검출에서 민감도를 증가.

* Houston et al

주 성분 분석으로 추가 수준의 분석을 사용.

이 절차에서 변형 normal equivalent image를 생성하는데, contral image 내에서 정상 편차를 설명하는데 도움을 준다.

주성분 분석이 있든지 없든지 간에, image coregistration 및 평균 통제 데이터셋을 사용한 해부학적 표준화는 임상과 진단 영상에서의 일상적 사용과 관련하여, 의심할 여지없이 확산되는 functional brain image의 객관적 해석에 있어 확실한 진전이다.

# Conclusion

의료 영상의 향상된 해석 및 분석을 위한 수많은 응용 프로그램에도 불구하고, image coregistration의 일성적인 임상 사용은 여전히 제한적이다.

질병 진행 평가 serial image의 해석을 개선하는 것에 분명한 이점이 존재한다.

이는 종양 및 다발성 경화증과 같은 질병에 대한 확립된, 잠재적 가입 또는 치료의 효과를 객관적으로 모니터링 해야하는 필요성을 포함한다.

* 다양한 ytpe의 image data(with intermodality coregistration)은 개별 양식 분석으로부터 구할 수 없는 정보를 얻을 수 있다.

* 뇌종양 재발의 치료와 수술을 위해 뇌전증을 국소화 하는데 있어 multimodality coregistration 사용의 이점이 있다.

* 수술 전 뇌전증 국소화는 가장 적은 침습적 절차(안전 & 효과적인 수술전략)의 결정은 전기생리학적 뇌전증 국소화를 뇌 매핑의 functional imaging에 결합하는 coregistration을 사용하는 것을 포함한다.

* intersubject coregistration은 일상적 임상 사용과 가장 멀지만, functional anatomic imaging의 정확한 정량적 해석을 위한 진정한 발전을 제공한다.

image coregistration의 적용이 느린 이유 : 지원의 어려움이 증가 하기 때문.

image coregistration을 이용하지 않는 이유 : 여러 다른 이미지 데이터의 자동화된 컴퓨터 조작을 효율적으로 수행하기 위한 logistic difficulty의 결과이다.


대부분 병원에서 고속 네트워크 구현 -> 디지털 이미지 아카이브의 빠른 접근에서 장애물이 제거됨

충분히 정확한, 검증된 coregistration은 대부분의 응용분야에 널리 사용

임상 의사및 영상 부서가 긴밀히 협력하고, 기존 이미지 처리 도구를 수집, 조정, 채택해야한다.

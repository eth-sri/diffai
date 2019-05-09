# Baseline
python . -D CIFAR10 -n ResNetTiny  -d "LinMix(a=Point(), b=Box(w=Lin(0,0.031373,150,10)), bw=Lin(0,0.5,150,10))"   --batch-size 50 --width 0.031373  --lr 0.001 --normalize-layer True --clip-norm False --lr-multistep $1
# InSamp
python . -D CIFAR10 -n ResNetTiny  -d "LinMix(a=Point(), b=InSamp(Lin(0,1,150,10)), bw=Lin(0,0.5, 150,10))" --batch-size 50 --width 0.031373  --lr 0.001 --normalize-layer True --clip-norm False --lr-multistep $1
# InSampLPA
python . -D CIFAR10 -n ResNetTiny  -d "LinMix(a=Point(), b=InSamp(Lin(0,1,150,20), w=Lin(0,0.031373, 150, 20)), bw=Lin(0,0.5, 150, 20))" --batch-size 50 --width 0.031373  --lr 0.001 --normalize-layer True --clip-norm False --lr-multistep  $1
# Adv_{1}InSampLPA
python . -D CIFAR10 -n ResNetTiny  -d "LinMix(a=IFGSM(w=Lin(0,0.031373,20,20), k=1), b=InSamp(Lin(0,1,150,10), w=Lin(0,0.031373,150,10)), bw=Lin(0,0.5,150,10))" --batch-size 50 --width 0.031373  --lr 0.001 --normalize-layer True --clip-norm False --lr-multistep $1
# Adv_{3}InSampLPA
python . -D CIFAR10 -n ResNetTiny  -d "LinMix(a=IFGSM(w=Lin(0,0.031373,20,20), k=3), b=InSamp(Lin(0,1,150,10), w=Lin(0,0.031373,150,10)), bw=Lin(0,0.5,150,10))" --batch-size 50 --width 0.031373  --lr 0.001  --normalize-layer True --clip-norm False --lr-multistep  $1


# Baseline
python . -D CIFAR10 -n ResNetTiny_FewCombo -d "LinMix(a=Point(), b=Box(w=Lin(0,0.031373,150,10)), bw=Lin(0,0.5,150,10))"   --batch-size 50 --width 0.031373  --lr 0.001 --normalize-layer True --clip-norm False --lr-multistep $1
# InSamp
python . -D CIFAR10 -n ResNetTiny_FewCombo -d "LinMix(a=Point(), b=InSamp(Lin(0,1,150,10)), bw=Lin(0,0.5, 150,10))" --batch-size 50 --width 0.031373  --lr 0.001 --normalize-layer True --clip-norm False --lr-multistep $1
# InSampLPA
python . -D CIFAR10 -n ResNetTiny_FewCombo -d "LinMix(a=Point(), b=InSamp(Lin(0,1,150,20), w=Lin(0,0.031373, 150, 20)), bw=Lin(0,0.5, 150, 20))" --batch-size 50 --width 0.031373  --lr 0.001 --normalize-layer True --clip-norm False --lr-multistep  $1
# Adv_{1}InSampLPA
python . -D CIFAR10 -n ResNetTiny_FewCombo -d "LinMix(a=IFGSM(w=Lin(0,0.031373,20,20), k=1), b=InSamp(Lin(0,1,150,10), w=Lin(0,0.031373,150,10)), bw=Lin(0,0.5,150,10))" --batch-size 50 --width 0.031373  --lr 0.001 --normalize-layer True --clip-norm False --lr-multistep $1
# Adv_{3}InSampLPA
python . -D CIFAR10 -n ResNetTiny_FewCombo -d "LinMix(a=IFGSM(w=Lin(0,0.031373,20,20), k=3), b=InSamp(Lin(0,1,150,10), w=Lin(0,0.031373,150,10)), bw=Lin(0,0.5,150,10))" --batch-size 50 --width 0.031373  --lr 0.001 --normalize-layer True --clip-norm False --lr-multistep  $1

# Adv_{1}InSampLPA
python . -D CIFAR10 -n ResNetTiny_ManyFixed -d "LinMix(a=IFGSM(w=Lin(0,0.031373,20,20), k=1), b=InSamp(Lin(0,1,150,10), w=Lin(0,0.031373,150,10)), bw=Lin(0,0.5,150,10))" --batch-size 50 --width 0.031373  --lr 0.001 --normalize-layer True --clip-norm False --lr-multistep $1

# InSamp_{18}
python . -D CIFAR10 -n SkipNet18 -d "LinMix(a=Point(), b=InSamp(Lin(0,1,200,40)), bw=Lin(0,0.5,200,40))" -t "MI_FGSM(k=20,r=2)" --batch-size 100 --save-freq 2 --width 0.031373 --lr 0.1 --normalize-layer True --clip-norm False --lr-multistep --sgd --custom-schedule "[10,20,250,300,350]"  $1
# Adv_{5}InSamp_{18}
python . -D CIFAR10 -n SkipNet18 -d "LinMix(a=IFGSM(w=Lin(0,0.031373,20,20)), b=InSamp(Lin(0,1,200,40)), bw=Lin(0,0.5,200,40))" -t "MI_FGSM(k=20,r=2)" --batch-size 100 --width 0.031373 --lr 0.1 --normalize-layer True --clip-norm False --lr-multistep --sgd --custom-schedule "[10,20,250,300,350]"  $1
# InSamp_{18}  Combo
python . -D CIFAR10 -n SkipNet18_Combo -d "LinMix(b=InSamp(Lin(0,1,200,40)), bw=Lin(0,0.5, 200, 40))" --batch-size 100 --width 0.031373 --lr 0.1 --normalize-layer True --clip-norm False --sgd --lr-multistep --custom-schedule "[10,20,250,300,350]" $1


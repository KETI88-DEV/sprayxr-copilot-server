# Task ID: 8

**Title:** Unsloth 환경 구축 및 QLoRA 학습

**Status:** done

**Dependencies:** 1 ✓, 7 ✓

**Priority:** high

**Description:** Docker 컨테이너 환경에서 Unsloth 2026.2.1 기반 QLoRA 학습, 4bit NF4 양자화, A40 메모리 최적화, 3-epoch 학습, adapter 저장

**Details:**

1. 모델 로딩: Unsloth FastLanguageModel로 Qwen2.5-7B-Instruct 4bit 로딩
   - load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
   - 모델은 /hf-cache/에서 자동 로드
2. LoRA 설정: rank=16, alpha=32, dropout=0.05
3. target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
4. 학습 하이퍼파라미터: batch_size=1 시작, grad_accum=8, max_seq_len=2048, optimizer=adamw_8bit, epochs=3
5. 안정성 단계: batch 1→2→4, seq 2048→4096, grad_accum 8→4로 점진적 상향
6. 학습 루프: SFTTrainer(Unsloth) 또는 manual training loop, gradient checkpointing
7. 저장: LoRA adapter를 /output/adapters/qwen25_shipcoat_lora/{adapter_config.json, adapter_model.bin}
   - 체크포인트 저장: /output/checkpoints/에 학습 중간 체크포인트 저장(epoch별)
8. 오류 처리: OOM 발생 시 즉시 batch 1 + seq 2048 + grad_accum 16으로 축소
9. 스크립트: /workspace/trainer/scripts/train_qlora.py, 설정 /workspace/trainer/configs/qlora_config.yaml
10. 모니터링: 로그 파일로 loss/accuracy 추적(/logs/train_qlora.log)

**Test Strategy:**

모델 로딩 성공(메모리 사용량 <40GB), 학습 시작 가능(첫 batch 처리 성공), adapter 저장 완료(/output/adapters/qwen25_shipcoat_lora/ 존재), loss curve 감소 추세 확인, 학습 시간 reasonable(<12시간)

## Subtasks

### 8.1. Unsloth 및 의존성 라이브러리 설치 및 검증

**Status:** done  
**Dependencies:** None  

Docker 컨테이너 환경에서 Unsloth 2026.2.1, bitsandbytes, peft, transformers 등 QLoRA 학습에 필요한 모든 라이브러리 설치 및 CUDA 호환성 검증

**Details:**

1. pip install unsloth 2026.2.1 및 의존성 패키지 설치
2. torch, bitsandbytes, peft, transformers 버전 호환성 확인
3. cuda available 및 A40 GPU 메모리(48GB) 인식 확인
4. import 테스트: unsloth, bitsandbytes, peft, transformers
5. FastLanguageModel 로딩 테스트(더미 모델 또는 소형 모델)
6. 설치 로그 기록(/logs/install_unsloth.log)

### 8.2. QLoRA 설정 YAML 파일 작성 및 파이썬 설정 검증

**Status:** done  
**Dependencies:** 8.1  

/workspace/trainer/configs/qlora_config.yaml 파일 생성 및 설정값(rank=16, alpha=32, batch_size, seq_len, epochs 등) 검증, 파이썬 config loader 구현

**Details:**

1. qlora_config.yaml 생성: lora_rank=16, lora_alpha=32, lora_dropout=0.05, target_modules 정의, batch_size=1, grad_accum=8, max_seq_len=2048, epochs=3, optimizer=adamw_8bit
2. 안정성 단계(scaling plan) 정의: batch 1→2→4, seq 2048→4096, grad_accum 8→4
3. 파이썬 YAML 로더 구현(dataclass 또는 dict)
4. 설정값 타입 및 범위 검증(batch_size>0, rank>0, alpha>0 등)
5. 설정 로드 테스트: python -c 'from trainer.config import load_qlora_config'
6. 로그 아웃풋으로 로드된 설정값 출력 확인

### 8.3. Qwen2.5-7B-Instruct 모델 로딩 및 4bit 양자화 테스트

**Status:** done  
**Dependencies:** 8.1, 8.2  

Unsloth FastLanguageModel을 사용하여 Qwen2.5-7B-Instruct 모델을 4bit NF4 양자화로 로드, /hf-cache/에서 자동 로드, 메모리 사용량 확인

**Details:**

1. FastLanguageModel.from_pretrained() 호출: model_name='Qwen/Qwen2.5-7B-Instruct', load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
2. HuggingFace cache 위치(/hf-cache/) 설정 확인
3. 4bit 양자화 메모리 사용량 측정(nvidia-smi 또는 torch.cuda.memory_allocated())
4. 메모리 사용량이 <40GB 이내 확인
5. 토크나이저 로드: AutoTokenizer.from_pretrained()
6. 더미 입력(test prompt)으로 forward pass 테스트
7. 로드 시간 및 메모리 사용량 로그 기록(/logs/model_load.log)

### 8.4. LoRA adapter 설정 및 준비, 초기 메모리 안정성 테스트

**Status:** done  
**Dependencies:** 8.3  

PEFT를 사용하여 LoRA adapter 설정(rank=16, alpha=32, dropout=0.05, target_modules 정의), 모델에 적용, 메모리 상태 점검 및 배치 크기 1에서 안정성 테스트

**Details:**

1. get_peft_model() 사용하여 LoRA adapter 추가: rank=16, lora_alpha=32, lora_dropout=0.05
2. target_modules 설정: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
3. adapter config 저장 준비(/output/adapters/qwen25_shipcoat_lora/ 디렉터리 생성)
4. 배치 크기 1, seq_len 2048, grad_accum 8 설정으로 더미 데이터(SFT 데이터 1건) 로드
5. 1 forward + backward pass 테스트
6. 메모리 피크 값 측정 및 기록
7. 메모리 부족 오류 없는지 확인, 있으면 조정 계획 기록

### 8.5. QLoRA 전체 학습 루프 구현 및 3-epoch 학습 실행

**Status:** done  
**Dependencies:** 8.4  

/workspace/trainer/scripts/train_qlora.py 작성: SFTTrainer 또는 manual training loop 구현, gradient checkpointing 포함, /data/sft/sft_train.jsonl 데이터 로드, 3-epoch 학습 완료, adapter 저장, loss curve 추적

**Details:**

1. 데이터 로드: /data/sft/sft_train.jsonl, DataLoader 구성(batch_size=1, shuffle=True)
2. SFTTrainer 초기화(권장) 또는 manual training loop 구현
   - gradient_checkpointing 활성화
   - optimizer=adamw_8bit, lr=2e-4, warmup_steps, lr_scheduler
   - torch.cuda.empty_cache() 주기적 호출
3. 안정성 단계 구현: epoch별로 batch_size 1→2→4 순서로 증가(또는 고정 유지)
4. 학습 루프: forward, backward, optimizer.step(), loss 기록
5. epoch별 평균 loss 계산 및 로그 기록(/logs/train_qlora.log)
6. OOM 발생 시 즉시 batch_size=1, seq_len=2048, grad_accum=16으로 축소하고 재시작 로직
7. 학습 완료 후 adapter 저장: /output/adapters/qwen25_shipcoat_lora/{adapter_config.json, adapter_model.bin}
8. 학습 시간 기록 및 loss curve 시각화(선택)

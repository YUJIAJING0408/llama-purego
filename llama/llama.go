package llama

import (
	"fmt"
	"syscall"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/jupiterrider/ffi"
)

type Token = int32
type SeqId = int32
type Pos = int32

// ---------- 精确对齐的结构体 ----------

type ModelParams struct {
	Devices             uintptr
	TensorBuftOverrides uintptr
	NGpuLayers          int32
	SplitMode           int32
	MainGpu             int32
	_                   [4]byte
	TensorSplit         uintptr
	ProgressCallback    uintptr
	ProgressCallbackUD  uintptr
	KVOverrides         uintptr
	VocabOnly           bool
	UseMmap             bool
	UseDirectIO         bool
	UseMlock            bool
	CheckTensors        bool
	UseExtraBufts       bool
	NoHost              bool
	NoAlloc             bool
}

type ContextParams struct {
	NCtx              uint32
	NBatch            uint32
	NUBatch           uint32
	NSeqMax           uint32
	NRsSeq            uint32
	NThreads          int32
	NThreadsBatch     int32
	CtxType           int32
	RopeScalingType   int32
	PoolingType       int32
	AttentionType     int32
	FlashAttnType     int32
	RopeFreqBase      float32
	RopeFreqScale     float32
	YarnExtFactor     float32
	YarnAttnFactor    float32
	YarnBetaFast      float32
	YarnBetaSlow      float32
	YarnOrigCtx       uint32
	DefragThold       float32
	CbEval            uintptr
	CbEvalUserData    uintptr
	TypeK             int32
	TypeV             int32
	AbortCallback     uintptr
	AbortCallbackData uintptr
	Embeddings        bool
	OffloadKQV        bool
	NoPerf            bool
	OpOffload         bool
	SwaFull           bool
	KVUnified         bool
	Samplers          uintptr
	NSamplers         uintptr
}

type Batch struct {
	NTokens int32
	Token   *Token
	Embd    *float32
	Pos     *Pos
	NSeqId  *int32
	SeqId   **SeqId
	Logits  *int8
}

// ---------- ffi 类型描述 ----------

var (
	typeModelParams   ffi.Type
	typeContextParams ffi.Type
	typeBatch         ffi.Type
)

func init() {
	typeModelParams = ffi.NewType(
		&ffi.TypePointer, &ffi.TypePointer,
		&ffi.TypeSint32, &ffi.TypeSint32, &ffi.TypeSint32,
		&ffi.TypeSint32,
		&ffi.TypePointer,
		&ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer,
		&ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeUint8,
		&ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeUint8,
	)

	typeContextParams = ffi.NewType(
		&ffi.TypeUint32, &ffi.TypeUint32, &ffi.TypeUint32, &ffi.TypeUint32, &ffi.TypeUint32,
		&ffi.TypeSint32, &ffi.TypeSint32, &ffi.TypeSint32, &ffi.TypeSint32, &ffi.TypeSint32,
		&ffi.TypeSint32, &ffi.TypeSint32,
		&ffi.TypeFloat, &ffi.TypeFloat, &ffi.TypeFloat, &ffi.TypeFloat,
		&ffi.TypeFloat, &ffi.TypeFloat, &ffi.TypeUint32, &ffi.TypeFloat,
		&ffi.TypePointer, &ffi.TypePointer, &ffi.TypeSint32, &ffi.TypeSint32,
		&ffi.TypePointer, &ffi.TypePointer,
		&ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeUint8, &ffi.TypeUint8,
		&ffi.TypeUint8, &ffi.TypeUint8,
		&ffi.TypePointer, &ffi.TypePointer,
	)

	typeBatch = ffi.NewType(
		&ffi.TypeSint32,
		&ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer,
		&ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer,
	)
}

// ---------- 全局变量 ----------

var (
	dllHandle uintptr

	addrModelDefaultParams        uintptr
	addrModelLoadFromFile         uintptr
	addrContextDefaultParams      uintptr
	addrInitFromModel             uintptr
	addrBatchInit                 uintptr
	addrDecode                    uintptr
	addrModelFree                 uintptr
	addrFree                      uintptr
	addrSamplerChainDefaultParams uintptr

	backendInit          func()
	modelGetVocab        func(model uintptr) uintptr
	vocabNTokens         func(vocab uintptr) int32
	vocabBOS             func(vocab uintptr) Token
	vocabEOS             func(vocab uintptr) Token
	tokenize             func(vocab uintptr, text *byte, textLen int32, tokens *Token, nTokensMax int32, addSpecial, parseSpecial bool) int32
	tokenToPiece         func(vocab uintptr, token Token, buf *byte, length int32, lstrip int32, special bool) int32
	getLogits            func(ctx uintptr) *float32
	samplerInitGreedy    func() uintptr
	samplerSample        func(smpl uintptr, ctx uintptr, idx int32) Token
	samplerFree          func(smpl uintptr)
	samplerInitTopK      func(int32) uintptr
	samplerInitTopP      func(float32, uintptr) uintptr
	samplerInitTemp      func(float32) uintptr
	samplerChainInit     func(unsafe.Pointer) uintptr
	samplerChainAdd      func(uintptr, uintptr)
	samplerInitDist      func(seed uint32) uintptr
	samplerInitMinP      func(float32, uintptr) uintptr
	samplerInitPenalties func(penaltyLastN int32, penaltyRepeat, penaltyFreq, penaltyPresent float32) uintptr
)

const LLAMA_DEFAULT_SEED uint32 = 0xFFFFFFFF

// ---------- 加载 DLL ----------

func LoadLibrary(libPath string) error {
	h, err := syscall.LoadLibrary(libPath)
	if err != nil {
		return fmt.Errorf("LoadLibrary: %w", err)
	}
	dllHandle = uintptr(h)

	getAddr := func(name string) uintptr {
		proc, err := syscall.GetProcAddress(syscall.Handle(dllHandle), name)
		if err != nil {
			panic(fmt.Sprintf("GetProcAddress(%s) failed: %v", name, err))
		}
		return uintptr(proc)
	}

	addrModelDefaultParams = getAddr("llama_model_default_params")
	addrModelLoadFromFile = getAddr("llama_model_load_from_file")
	addrContextDefaultParams = getAddr("llama_context_default_params")
	addrInitFromModel = getAddr("llama_init_from_model")
	addrBatchInit = getAddr("llama_batch_init")
	addrDecode = getAddr("llama_decode")
	addrModelFree = getAddr("llama_model_free")
	addrFree = getAddr("llama_free")
	addrSamplerChainDefaultParams = getAddr("llama_sampler_chain_default_params")

	purego.RegisterLibFunc(&backendInit, dllHandle, "llama_backend_init")
	purego.RegisterLibFunc(&modelGetVocab, dllHandle, "llama_model_get_vocab")
	purego.RegisterLibFunc(&vocabNTokens, dllHandle, "llama_vocab_n_tokens")
	purego.RegisterLibFunc(&vocabBOS, dllHandle, "llama_vocab_bos")
	purego.RegisterLibFunc(&vocabEOS, dllHandle, "llama_vocab_eos")
	purego.RegisterLibFunc(&tokenize, dllHandle, "llama_tokenize")
	purego.RegisterLibFunc(&tokenToPiece, dllHandle, "llama_token_to_piece")
	purego.RegisterLibFunc(&getLogits, dllHandle, "llama_get_logits")
	purego.RegisterLibFunc(&samplerInitGreedy, dllHandle, "llama_sampler_init_greedy")
	purego.RegisterLibFunc(&samplerSample, dllHandle, "llama_sampler_sample")
	purego.RegisterLibFunc(&samplerFree, dllHandle, "llama_sampler_free")
	purego.RegisterLibFunc(&samplerInitTopK, dllHandle, "llama_sampler_init_top_k")
	purego.RegisterLibFunc(&samplerInitTopP, dllHandle, "llama_sampler_init_top_p")
	purego.RegisterLibFunc(&samplerInitTemp, dllHandle, "llama_sampler_init_temp")
	purego.RegisterLibFunc(&samplerChainInit, dllHandle, "llama_sampler_chain_init")
	purego.RegisterLibFunc(&samplerChainAdd, dllHandle, "llama_sampler_chain_add")
	purego.RegisterLibFunc(&samplerInitDist, dllHandle, "llama_sampler_init_dist")
	purego.RegisterLibFunc(&samplerInitMinP, dllHandle, "llama_sampler_init_min_p")
	purego.RegisterLibFunc(&samplerInitPenalties, dllHandle, "llama_sampler_init_penalties")

	return nil
}

// 添加辅助函数 defaultSamplerChainParams
func defaultSamplerChainParams() bool {
	ptr := call0(addrSamplerChainDefaultParams, &ffi.TypeUint8)
	return *(*bool)(ptr)
}

func BackendInit() { backendInit() }

// ---------- 辅助 ffi 调用 ----------

// call0 无参，返回结构体（通过调用者分配的空间返回）
func call0(fn uintptr, rtype *ffi.Type) unsafe.Pointer {
	var cif ffi.Cif
	if st := ffi.PrepCif(&cif, ffi.DefaultAbi, 0, rtype); st != ffi.OK {
		panic("PrepCif failed")
	}
	buf := make([]byte, rtype.Size)
	ffi.Call(&cif, fn, unsafe.Pointer(&buf[0]))
	return unsafe.Pointer(&buf[0])
}

// call2ptr 两个参数 (const char*, struct by value)，返回 pointer
func call2ptr(fn uintptr, path string, params unsafe.Pointer, paramType *ffi.Type) uintptr {
	cPath, err := syscall.BytePtrFromString(path)
	if err != nil {
		panic(err)
	}
	var cif ffi.Cif
	if st := ffi.PrepCif(&cif, ffi.DefaultAbi, 2, &ffi.TypePointer, &ffi.TypePointer, paramType); st != ffi.OK {
		panic("PrepCif failed")
	}
	ptrVar := unsafe.Pointer(cPath)
	var ret uintptr
	ffi.Call(&cif, fn, unsafe.Pointer(&ret),
		unsafe.Pointer(&ptrVar),
		params,
	)
	return ret
}

// call2ctx 两个参数 (model ptr, struct by value)，返回 pointer
func call2ctx(fn uintptr, model uintptr, params *ContextParams) uintptr {
	var cif ffi.Cif
	if st := ffi.PrepCif(&cif, ffi.DefaultAbi, 2, &ffi.TypePointer, &ffi.TypePointer, &typeContextParams); st != ffi.OK {
		panic("PrepCif failed")
	}
	modelPtr := unsafe.Pointer(model)
	var ret uintptr
	ffi.Call(&cif, fn, unsafe.Pointer(&ret),
		unsafe.Pointer(&modelPtr),
		unsafe.Pointer(params),
	)
	return ret
}

// callDecode ctx(pointer) + batch(struct)
func callDecode(fn uintptr, ctx uintptr, batch Batch) int32 {
	var cif ffi.Cif
	if st := ffi.PrepCif(&cif, ffi.DefaultAbi, 2, &ffi.TypeSint32, &ffi.TypePointer, &typeBatch); st != ffi.OK {
		panic("PrepCif failed")
	}
	ctxPtr := unsafe.Pointer(ctx)
	var ret int32
	ffi.Call(&cif, fn, unsafe.Pointer(&ret),
		unsafe.Pointer(&ctxPtr),
		unsafe.Pointer(&batch),
	)
	return ret
}

// callFree 一个指针参数，无返回
func callFree(fn uintptr, ptr uintptr) {
	var cif ffi.Cif
	if st := ffi.PrepCif(&cif, ffi.DefaultAbi, 1, &ffi.TypeVoid, &ffi.TypePointer); st != ffi.OK {
		panic("PrepCif failed")
	}
	ptrVal := unsafe.Pointer(ptr)
	ffi.Call(&cif, fn, nil, unsafe.Pointer(&ptrVal))
}

// ---------- 公开 API ----------

type Model struct {
	ptr   uintptr
	vocab uintptr
}

type Context struct {
	ptr     uintptr
	vocab   uintptr
	sampler uintptr
}

func DefaultModelParams() ModelParams {
	return *(*ModelParams)(call0(addrModelDefaultParams, &typeModelParams))
}

func DefaultContextParams() ContextParams {
	return *(*ContextParams)(call0(addrContextDefaultParams, &typeContextParams))
}

func LoadModel(path string) (*Model, error) {
	params := DefaultModelParams()
	ptr := call2ptr(addrModelLoadFromFile, path, unsafe.Pointer(&params), &typeModelParams)
	if ptr == 0 {
		return nil, fmt.Errorf("failed to load model: %s", path)
	}
	vocab := modelGetVocab(ptr)
	return &Model{ptr: ptr, vocab: vocab}, nil
}

func (m *Model) Free() {
	callFree(addrModelFree, m.ptr)
}

func NewSamplerChain() uintptr {
	noPerf := defaultSamplerChainParams()
	var params byte
	if noPerf {
		params = 1
	}
	chain := samplerChainInit(unsafe.Pointer(&params))

	// 1. 惩罚层 (Penalties)
	samplerChainAdd(chain, samplerInitPenalties(64, 1.2, 0.0, 0.0))

	// 2. Top-K 过滤
	samplerChainAdd(chain, samplerInitTopK(30))

	// 3. Top-P 过滤
	samplerChainAdd(chain, samplerInitTopP(0.9, 1))

	// 4. Min-P 过滤
	samplerChainAdd(chain, samplerInitMinP(0.1, 1))

	// 5. 温度调节
	samplerChainAdd(chain, samplerInitTemp(0.6))

	// 6. 最终选择器
	samplerChainAdd(chain, samplerInitDist(LLAMA_DEFAULT_SEED))

	return chain
}

func NewContext(model *Model) (*Context, error) {
	params := DefaultContextParams()
	params.NCtx = 4096
	params.Embeddings = false
	ptr := call2ctx(addrInitFromModel, model.ptr, &params)
	if ptr == 0 {
		return nil, fmt.Errorf("failed to create context")
	}
	// 关键修改：使用多级采样链
	sampler := NewSamplerChain()
	return &Context{ptr: ptr, vocab: model.vocab, sampler: sampler}, nil
}

func (c *Context) Free() {
	samplerFree(c.sampler)
	callFree(addrFree, c.ptr)
}

func (c *Context) Tokenize(text string, addSpecial bool) []Token {
	tokens := make([]Token, len(text)+32)
	cStr, _ := syscall.BytePtrFromString(text)
	n := tokenize(c.vocab, cStr, int32(len(text)), &tokens[0], int32(len(tokens)), addSpecial, false)
	if n < 0 {
		tokens = make([]Token, -n)
		n = tokenize(c.vocab, cStr, int32(len(text)), &tokens[0], int32(len(tokens)), addSpecial, false)
	}
	return tokens[:n]
}

func (c *Context) TokenToPiece(token Token) string {
	buf := make([]byte, 64)
	n := tokenToPiece(c.vocab, token, &buf[0], int32(len(buf)), 0, false)
	if n < 0 {
		buf = make([]byte, -n)
		n = tokenToPiece(c.vocab, token, &buf[0], int32(len(buf)), 0, false)
	}
	return string(buf[:n])
}

func (c *Context) Decode(batch Batch) error {
	ret := callDecode(addrDecode, c.ptr, batch)
	if ret != 0 {
		return fmt.Errorf("decode error: %d", ret)
	}
	return nil
}

func (c *Context) GetLogits() []float32 {
	nVocab := vocabNTokens(c.vocab)
	logitsPtr := getLogits(c.ptr)
	return unsafe.Slice(logitsPtr, nVocab)
}

func (c *Context) Sample() Token {
	return samplerSample(c.sampler, c.ptr, -1)
}

func (c *Context) EOS() Token {
	return vocabEOS(c.vocab)
}

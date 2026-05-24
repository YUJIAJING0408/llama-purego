package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/YUJIAJING0408/llama-purego/llama"
)

func main() {
	if err := llama.LoadLibrary("llama.dll"); err != nil {
		panic(err)
	}
	fmt.Println("✅ DLL 加载成功")

	llama.BackendInit()
	fmt.Println("✅ 后端初始化完成")

	model, err := llama.LoadModel("I:\\Codes\\Go\\llama-purego\\models\\Qwen3.5-0.8B-Q4_0.gguf")
	if err != nil {
		panic(err)
	}
	defer model.Free()
	fmt.Println("✅ 模型加载完成")

	ctx, err := llama.NewContext(model)
	if err != nil {
		panic(err)
	}
	defer ctx.Free()
	fmt.Println("✅ 上下文创建完成")

	fmt.Println("\n🤖 输入 '退出' 结束对话")
	scanner := bufio.NewScanner(os.Stdin)
	for fmt.Print("\n👤 你: "); scanner.Scan(); fmt.Print("\n👤 你: ") {
		input := strings.TrimSpace(scanner.Text())
		if input == "退出" {
			break
		}
		if input == "" {
			continue
		}
		fmt.Print("🤖 助手: ")
		generate(ctx, input)
	}
}

func generate(ctx *llama.Context, userInput string) {
	systemPrompt := "你是一个AI个人助手"
	prompt := fmt.Sprintf("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", systemPrompt, userInput)
	tokens := ctx.Tokenize(prompt, false) // addSpecial = false，因为我们已经手动添加特殊token
	if len(tokens) == 0 {
		fmt.Println("(无输入)")
		return
	}

	nTokens := int32(len(tokens))
	logitsMask := make([]int8, nTokens)
	logitsMask[nTokens-1] = 1

	batch := llama.Batch{
		NTokens: nTokens,
		Token:   &tokens[0],
		Logits:  &logitsMask[0],
	}
	if err := ctx.Decode(batch); err != nil {
		fmt.Printf("\n[解码错误] %v\n", err)
		return
	}

	eos := ctx.EOS()

	for {
		token := ctx.Sample()
		if token == eos {
			break
		}
		piece := ctx.TokenToPiece(token)
		// 关键检查：如果模型输出了 "<|im_start|>" 或 "<|im_end|>"（可能是 EOS 但已被上面拦截），立即停止
		// 因为 "<|im_end|>" 就是 EOS，所以这里主要防止 "<|im_start|>"
		if len(piece) > 0 && (piece[0] == '<' && strings.HasPrefix(piece, "<|im_start|>")) {
			break
		}
		fmt.Print(piece)

		nextTokens := []llama.Token{token}
		batch = llama.Batch{
			NTokens: 1,
			Token:   &nextTokens[0],
			Logits:  &[]int8{1}[0],
		}
		if err := ctx.Decode(batch); err != nil {
			fmt.Printf("\n[解码错误] %v\n", err)
			return
		}
	}
	fmt.Println()
}

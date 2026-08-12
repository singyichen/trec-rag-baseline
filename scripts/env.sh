# 共用環境設定,由其他腳本 source,不直接執行。

# 專案有 .venv 而目前 shell 尚未啟用任何 venv 時,自動啟用,
# 避免誤用系統 Python(那裡沒裝 pyserini)。
if [[ -z "${VIRTUAL_ENV:-}" && -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

# pyserini 1.6 在 import 時會初始化 OpenAI client,沒設環境變數會直接報錯;
# 我們不用它的 LLM 功能,占位值即可。
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

# macOS:pyjnius 靠 JAVA_HOME 找 JVM(不看 PATH)。
# Homebrew 的 openjdk 是 keg-only,若尚未註冊到 /Library/Java/JavaVirtualMachines,
# 這裡直接退回 Homebrew 的安裝路徑。
if [[ "$(uname)" == "Darwin" && -z "${JAVA_HOME:-}" ]]; then
  JAVA_HOME="$(/usr/libexec/java_home -v 21 2>/dev/null || true)"
  if [[ -z "$JAVA_HOME" ]]; then
    for p in /opt/homebrew/opt/openjdk@21 /usr/local/opt/openjdk@21; do
      if [[ -d "$p/libexec/openjdk.jdk/Contents/Home" ]]; then
        JAVA_HOME="$p/libexec/openjdk.jdk/Contents/Home"
        break
      fi
    done
  fi
  if [[ -n "$JAVA_HOME" ]]; then
    export JAVA_HOME
    export PATH="$JAVA_HOME/bin:$PATH"
  else
    echo "找不到 Java 21。請先:brew install openjdk@21" >&2
    echo "並註冊:sudo ln -sfn /opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-21.jdk" >&2
    exit 1
  fi
fi

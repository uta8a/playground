mod greeting;
use greeting::exports::anonymous::greeting::say;

struct Component;

/// witの`interface say`で宣言した関数を実装する
/// 関数の列挙は`Rust-Analyzer`に任せるのが楽
impl say::Guest for Component {
    // _rt::Stringで作成されるけど単なるstd::string::Stringのエイリアス
    fn hello() -> String {
        "Hello World !!".into()
    }
}

// コンポーネントの場合必ず最後にexportマクロを呼ぶ
greeting::export!(Component with_types_in greeting);

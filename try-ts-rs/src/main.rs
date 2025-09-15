use ts_rs::TS;

#[derive(TS)]
#[ts(export)]
struct User {
    user_id: i32,
    first_name: String,
    last_name: String,
}

#[derive(TS)]
// when 'serde-compat' is enabled, ts-rs tries to use supported serde attributes.
#[ts(export)]
enum Gender {
    Male,
    Female,
    Other,
}

fn main() {
    println!("Hello, world!");
}

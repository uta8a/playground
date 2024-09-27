import { assertEquals } from "@std/assert";
import { parse } from "./main.ts";

const DUMMY_DATA = `# 記事のタイトル
https://hoge.example.com/
記事の内容解説

# 記事のタイトル2: hoge
https://fuga.example.com/
記事の内容解説2
`;

// Ty: Articleからid, created_atを除外したもの
const EXPECTED_NOT_INCLUDE_ID_CREATED_AT = [
  {
    filename: "記事のタイトル",
    frontmatter: {
      url: "https://hoge.example.com/",
      title: "記事のタイトル",
      acl: "private",
    },
    content: "記事の内容解説",
  },
  {
    filename: "記事のタイトル2__hoge",
    frontmatter: {
      url: "https://fuga.example.com/",
      title: "記事のタイトル2: hoge",
      acl: "private",
    },
    content: "記事の内容解説2",
  },
];

Deno.test(function parse_article_data_static() {
  const parsed = parse(DUMMY_DATA).map((article) => ({
    filename: article.filename,
    frontmatter: {
      url: article.frontmatter.url,
      title: article.frontmatter.title,
      acl: article.frontmatter.acl,
    },
    content: article.content,
  }));
  // id, created_at は動的な値なので、比較対象から除外する
  const actual = parsed.map((article) => ({
    filename: article.filename,
    frontmatter: {
      url: article.frontmatter.url,
      title: article.frontmatter.title,
      acl: article.frontmatter.acl,
    },
    content: article.content,
  }));
  assertEquals(parsed, EXPECTED_NOT_INCLUDE_ID_CREATED_AT);
});

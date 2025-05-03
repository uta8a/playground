import { createTool } from "@mastra/core/tools";
import { z } from "zod";
import fetch from "node-fetch";

// 有効な言語コードのリスト
const validLanguages = [
  'ar', 'eu', 'bn', 'bg', 'ca', 'zh-hans', 'zh-hant', 'hr', 'cs', 'da', 'nl', 'en', 'en-gb', 
  'et', 'fi', 'fr', 'gl', 'de', 'gu', 'he', 'hi', 'hu', 'is', 'it', 'jp', 'kn', 'ko', 'lv', 
  'lt', 'ms', 'ml', 'mr', 'nb', 'pl', 'pt-br', 'pt-pt', 'pa', 'ro', 'ru', 'sr', 'sk', 'sl', 
  'es', 'sv', 'ta', 'te', 'th', 'tr', 'uk', 'vi'
] as const;

export const webSearchTool = createTool({
  id: "web-search",
  description: "検索エンジンを使用してウェブ検索を実行します",
  inputSchema: z.object({
    query: z.string().describe("検索クエリ"),
    country: z.string().optional().describe("検索結果の国コード（例: JP, US）"),
    count: z.number().optional().describe("返される検索結果の最大数（デフォルト: 10）"),
    language: z.enum(validLanguages).optional()
      .describe("検索言語（例: jp=日本語, en=英語）。使用可能な値: " + validLanguages.join(', ')),
    offset: z.number().optional().describe("検索結果のオフセット"),
  }),
  outputSchema: z.object({
    results: z.array(
      z.object({
        title: z.string(),
        url: z.string(),
        description: z.string(),
      })
    ),
    query: z.string(),
    total_results: z.number().optional(),
  }),
  execute: async ({ context }: { context: { query: string; country?: string; count?: number; language?: string; offset?: number } }) => {
    return await performWebSearch(
      context.query,
      context.country,
      context.count,
      context.language,
      context.offset
    );
  },
});

interface WebSearchResponse {
  type: string;
  web?: {
    results: Array<{
      title: string;
      url: string;
      description: string;
    }>;
    total_results?: number;
  };
  query?: {
    original: string;
  };
}

const performWebSearch = async (
  query: string,
  country?: string,
  count?: number,
  language?: string,
  offset?: number
) => {
  // APIキーは環境変数から取得
  const apiKey = process.env.BRAVE_API_KEY;
  if (!apiKey) {
    throw new Error("Brave Search APIキーが設定されていません。環境変数 BRAVE_API_KEY を設定してください。");
  }

  const baseUrl = "https://api.search.brave.com/res/v1/web/search";

  // URLパラメータの構築
  const params = new URLSearchParams({
    q: query
  });
  
  // オプションパラメータの追加
  if (country) params.append("country", country);
  if (count) params.append("count", count.toString());
  if (language) params.append("search_lang", language);
  if (offset) params.append("offset", offset.toString());

  const url = `${baseUrl}?${params.toString()}`;

  try {
    const response = await fetch(url, {
      method: "GET",
      headers: {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": apiKey
      }
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Brave API エラー: ${response.status} ${errorText}`);
    }

    const data = await response.json() as WebSearchResponse;

    // レスポンスの整形
    return {
      results: data.web?.results.map(result => ({
        title: result.title,
        url: result.url,
        description: result.description
      })) || [],
      query: data.query?.original || query,
      total_results: data.web?.total_results
    };
  } catch (error) {
    console.error("Web search error:", error);
    const errorMessage = error instanceof Error ? error.message : String(error);
    throw new Error(`検索の実行中にエラーが発生しました: ${errorMessage}`);
  }
};

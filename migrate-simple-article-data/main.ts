import { v4 } from "https://deno.land/std@0.106.0/uuid/mod.ts";

type Article = {
    filename: string;
    frontmatter: {
        id: string;
        url: string;
        title: string;
        created_at: string;
        acl: string;
    };
    content: string;
};

const adjustFilename = (str: string): string => {
    return str.replace(/[^a-zA-Z0-9\u3040-\u30FF\u4E00-\u9FFF]/g, "_")
        .toLowerCase();
};

const parse = (data: string): Article[] => {
    const rawData = data.split(/^\# /gm);
    const articles: Article[] = rawData.map((raw) => {
        const lines = raw.split("\n");
        if (lines.length < 3) {
            return undefined;
        }
        return {
            filename: adjustFilename(lines[0]), // modify adjust filename
            frontmatter: {
                id: v4.generate(),
                url: lines[1],
                title: lines[0], // 確定
                created_at: new Date().toISOString(),
                acl: "private",
            },
            content: lines.slice(2).join("\n").trim(),
        };
    }).filter((article) => article !== undefined);
    return articles;
};

const print = (articles: Article[]): Promise<void> => {
    // Article to write file
    for (const article of articles) {
        const filename = `./out/${article.filename}.md`;
        const content = `---
id: "${article.frontmatter.id}"
url: "${article.frontmatter.url}"
title: "${article.frontmatter.title}"
created_at: "${article.frontmatter.created_at}"
acl: "${article.frontmatter.acl}"
---

#reading

` + article.content;
        Deno.writeTextFile(filename, content);
    }
    return Promise.resolve();
};

await print(parse(Deno.readTextFileSync("./data.md")));

export { parse };

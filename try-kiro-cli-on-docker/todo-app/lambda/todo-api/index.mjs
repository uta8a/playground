import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import {
  DynamoDBDocumentClient,
  ScanCommand,
  PutCommand,
  UpdateCommand,
  DeleteCommand,
} from "@aws-sdk/lib-dynamodb";
import { randomUUID } from "crypto";

const client = DynamoDBDocumentClient.from(new DynamoDBClient({}));
const TABLE = process.env.TABLE_NAME;

const res = (status, body) => ({
  statusCode: status,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const handler = async (event) => {
  const method = event.requestContext.http.method;
  const id = event.pathParameters?.id;

  if (method === "GET") {
    const { Items } = await client.send(new ScanCommand({ TableName: TABLE }));
    return res(200, Items);
  }

  if (method === "POST") {
    const { title } = JSON.parse(event.body || "{}");
    if (!title) return res(400, { error: "title required" });
    const now = new Date().toISOString();
    const item = { id: randomUUID(), title, completed: false, createdAt: now, updatedAt: now };
    await client.send(new PutCommand({ TableName: TABLE, Item: item }));
    return res(201, item);
  }

  if (method === "PATCH" && id) {
    const { title, completed } = JSON.parse(event.body || "{}");
    const updates = [];
    const names = {};
    const values = { ":u": new Date().toISOString() };
    if (title !== undefined) { updates.push("#t = :t"); names["#t"] = "title"; values[":t"] = title; }
    if (completed !== undefined) { updates.push("#c = :c"); names["#c"] = "completed"; values[":c"] = completed; }
    updates.push("updatedAt = :u");
    const { Attributes } = await client.send(new UpdateCommand({
      TableName: TABLE, Key: { id },
      UpdateExpression: "SET " + updates.join(", "),
      ExpressionAttributeNames: Object.keys(names).length ? names : undefined,
      ExpressionAttributeValues: values,
      ReturnValues: "ALL_NEW",
    }));
    return res(200, Attributes);
  }

  if (method === "DELETE" && id) {
    await client.send(new DeleteCommand({ TableName: TABLE, Key: { id } }));
    return res(204, null);
  }

  return res(404, { error: "Not found" });
};

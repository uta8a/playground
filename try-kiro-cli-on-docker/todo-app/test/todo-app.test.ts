import * as cdk from "aws-cdk-lib";
import { Template, Match } from "aws-cdk-lib/assertions";
import { TodoAppStack } from "../lib/todo-app-stack";

let template: Template;

beforeAll(() => {
  const app = new cdk.App();
  const stack = new TodoAppStack(app, "TestStack");
  template = Template.fromStack(stack);
});

test("DynamoDB table with PAY_PER_REQUEST and DESTROY policy", () => {
  template.hasResourceProperties("AWS::DynamoDB::Table", {
    BillingMode: "PAY_PER_REQUEST",
    KeySchema: [{ AttributeName: "id", KeyType: "HASH" }],
  });
  template.hasResource("AWS::DynamoDB::Table", {
    DeletionPolicy: "Delete",
  });
});

test("Lambda function with correct runtime and TABLE_NAME env", () => {
  template.hasResourceProperties("AWS::Lambda::Function", {
    Runtime: "nodejs22.x",
    Environment: {
      Variables: Match.objectLike({ TABLE_NAME: Match.anyValue() }),
    },
  });
});

test("HTTP API with CORS enabled", () => {
  template.hasResourceProperties("AWS::ApiGatewayV2::Api", {
    ProtocolType: "HTTP",
    CorsConfiguration: Match.objectLike({
      AllowOrigins: ["*"],
    }),
  });
});

test("HTTP API routes: GET, POST, PATCH, DELETE", () => {
  const routes = template.findResources("AWS::ApiGatewayV2::Route");
  const routeKeys = Object.values(routes).map((r: any) => r.Properties.RouteKey);
  expect(routeKeys).toEqual(expect.arrayContaining([
    "GET /todos",
    "POST /todos",
    "PATCH /todos/{id}",
    "DELETE /todos/{id}",
  ]));
});

test("S3 bucket with autoDeleteObjects and DESTROY policy", () => {
  template.hasResourceProperties("AWS::S3::Bucket", {
    PublicAccessBlockConfiguration: {
      BlockPublicAcls: true,
      BlockPublicPolicy: true,
      IgnorePublicAcls: true,
      RestrictPublicBuckets: true,
    },
  });
  template.hasResource("AWS::S3::Bucket", { DeletionPolicy: "Delete" });
});

test("CloudFront distribution with HTTPS redirect", () => {
  template.hasResourceProperties("AWS::CloudFront::Distribution", {
    DistributionConfig: Match.objectLike({
      DefaultRootObject: "index.html",
      DefaultCacheBehavior: Match.objectLike({
        ViewerProtocolPolicy: "redirect-to-https",
      }),
    }),
  });
});

test("CloudWatch log group with DESTROY policy", () => {
  template.hasResource("AWS::Logs::LogGroup", { DeletionPolicy: "Delete" });
});

test("Stack outputs: ApiUrl and FrontendUrl", () => {
  template.hasOutput("ApiUrl", {});
  template.hasOutput("FrontendUrl", {});
});

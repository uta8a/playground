import { Construct } from "constructs";
import { Duration, RemovalPolicy } from "aws-cdk-lib";
import { Function, Runtime, Code } from "aws-cdk-lib/aws-lambda";
import { HttpApi, HttpMethod, CorsHttpMethod } from "aws-cdk-lib/aws-apigatewayv2";
import { HttpLambdaIntegration } from "aws-cdk-lib/aws-apigatewayv2-integrations";

import { Table } from "aws-cdk-lib/aws-dynamodb";
import { LogGroup, RetentionDays } from "aws-cdk-lib/aws-logs";
import * as path from "path";

export class ApiConstruct extends Construct {
  readonly apiUrl: string;

  constructor(scope: Construct, id: string, table: Table) {
    super(scope, id);

    const functionName = "todo-api-handler";

    const logGroup = new LogGroup(this, "LambdaLogs", {
      logGroupName: `/aws/lambda/${functionName}`,
      retention: RetentionDays.ONE_DAY,
      removalPolicy: RemovalPolicy.DESTROY,
    });

    const fn = new Function(this, "TodoApiHandler", {
      functionName: functionName,
      runtime: Runtime.NODEJS_22_X,
      handler: "index.handler",
      code: Code.fromAsset(path.join(__dirname, "../../lambda/todo-api")),
      environment: { TABLE_NAME: table.tableName },
      timeout: Duration.seconds(10),
      logGroup: logGroup,
    });

    fn.node.addDependency(logGroup);

    table.grantReadWriteData(fn);

    const integration = new HttpLambdaIntegration("TodoApiLambdaIntegration", fn);

    const api = new HttpApi(this, "TodoApi", {
      corsPreflight: {
        allowOrigins: ["*"],
        allowMethods: [CorsHttpMethod.ANY],
        allowHeaders: ["Content-Type"],
      },
    });

    for (const [method, path] of [
      [HttpMethod.GET, "/todos"],
      [HttpMethod.POST, "/todos"],
      [HttpMethod.PATCH, "/todos/{id}"],
      [HttpMethod.DELETE, "/todos/{id}"],
    ] as [HttpMethod, string][]) {
      api.addRoutes({ path, methods: [method], integration });
    }

    this.apiUrl = api.url!;
  }
}

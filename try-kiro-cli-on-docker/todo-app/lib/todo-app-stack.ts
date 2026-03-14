import { Stack, StackProps, CfnOutput } from "aws-cdk-lib";
import { Construct } from "constructs";
import { DatabaseConstruct } from "./constructs/database";
import { ApiConstruct } from "./constructs/api";
import { FrontendConstruct } from "./constructs/frontend";

export class TodoAppStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    const db = new DatabaseConstruct(this, "Database");
    const api = new ApiConstruct(this, "Api", db.table);
    new FrontendConstruct(this, "Frontend", api.apiUrl);

    new CfnOutput(this, "ApiUrl", { value: api.apiUrl });
  }
}

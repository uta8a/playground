import { Construct } from "constructs";
import { RemovalPolicy, CfnOutput } from "aws-cdk-lib";
import { Bucket, BlockPublicAccess } from "aws-cdk-lib/aws-s3";
import { BucketDeployment, Source } from "aws-cdk-lib/aws-s3-deployment";
import {
  Distribution,
  ViewerProtocolPolicy,
  CachePolicy,
} from "aws-cdk-lib/aws-cloudfront";
import { S3BucketOrigin } from "aws-cdk-lib/aws-cloudfront-origins";
import * as path from "path";

export class FrontendConstruct extends Construct {
  constructor(scope: Construct, id: string, apiUrl: string) {
    super(scope, id);

    const bucket = new Bucket(this, "FrontendBucket", {
      blockPublicAccess: BlockPublicAccess.BLOCK_ALL,
      autoDeleteObjects: true,
      removalPolicy: RemovalPolicy.DESTROY,
    });

    const distribution = new Distribution(this, "Distribution", {
      defaultBehavior: {
        origin: S3BucketOrigin.withOriginAccessControl(bucket),
        viewerProtocolPolicy: ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
        cachePolicy: CachePolicy.CACHING_DISABLED,
      },
      defaultRootObject: "index.html",
    });

    new BucketDeployment(this, "Deploy", {
      sources: [
        Source.asset(path.join(__dirname, "../../frontend")),
        Source.jsonData("config.json", { apiUrl }),
      ],
      destinationBucket: bucket,
      distribution,
    });

    new CfnOutput(this, "FrontendUrl", {
      value: `https://${distribution.distributionDomainName}`,
    });
  }
}

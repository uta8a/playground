---
name: cdk-list-resources
description: List all AWS resources that will be created by a CDK stack. Use when asked to show, summarize, or audit the resources defined in a CDK app, including resource types, logical IDs, and physical names where determinable.
---

# CDK Resource Lister

## Overview
Enumerate all AWS resources a CDK stack will create by running `cdk synth` and parsing the CloudFormation template output.

## Workflow
1. Run `cdk synth` in the CDK project root to generate the CloudFormation template.
2. Parse the `Resources` section of the template.
3. Group resources by AWS service and present them in a readable table.

## Steps

### 1. Synthesize
```bash
npx cdk synth 2>/dev/null
```
If multiple stacks exist, list them first with `npx cdk list` and synthesize the target stack.

### 2. Extract Resources
From the synthesized template, extract each entry under `Resources`:
- `Type` — CloudFormation resource type (e.g. `AWS::DynamoDB::Table`)
- Logical ID — the key in the `Resources` map
- Physical name — from `Properties.BucketName`, `Properties.TableName`, `Properties.FunctionName`, etc. if explicitly set; otherwise note it is auto-generated.

### 3. Output Format
Present results grouped by service:

```
## Resources: <StackName>

### DynamoDB
| Logical ID | Type | Name |
|---|---|---|
| ... | AWS::DynamoDB::Table | auto-generated |

### Lambda
...
```

Include a summary line at the end:
```
Total: N resources across M services
```

## Notes
- Resources with `AWS::CDK::Metadata` or `CDKMetadata` type should be excluded from the list.
- For auto-generated names, state "auto-generated (deploy-time)" rather than guessing.
- If `cdk synth` fails, report the error and suggest running `npm install` or checking `cdk.json`.

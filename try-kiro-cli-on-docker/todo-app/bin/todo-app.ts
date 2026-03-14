#!/usr/bin/env node
import "source-map-support/register";
import * as cdk from "aws-cdk-lib";
import { TodoAppStack } from "../lib/todo-app-stack";

const app = new cdk.App();
const stack = new TodoAppStack(app, "TodoApp");

cdk.Tags.of(stack).add("Project", "todo-app");
cdk.Tags.of(stack).add("ManagedBy", "cdk");
cdk.Tags.of(stack).add("Environment", app.node.tryGetContext("env") ?? "dev");

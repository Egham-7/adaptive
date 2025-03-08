import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/_home/conversations/$conversationId')({
  component: RouteComponent,
})

function RouteComponent() {
  return <div>Hello "/_home/conversations/$conversationId"!</div>
}

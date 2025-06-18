export default async function (input) {
  const response = await fetch("https://web-production-91846.up.railway.app/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: input }),
  });

  const json = await response.json();
  return json.answer;
}

// Temporary manual test — remove this when done
if (import.meta.url === `file://${process.argv[1]}`) {
  const answer = await (await import('./virtual-ta-provider.mjs')).default("What is the tools layer?");
  console.log("Response:", answer);
}

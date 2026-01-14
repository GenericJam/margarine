# Generate a logo for Margarine
prompt = "A single rectangular stick of yellow margarine sitting in a spreading puddle of melted butter. The margarine stick is golden yellow with smooth rounded corners, slightly melting and dripping down the sides. Below it is a large irregular puddle of liquid margarine spreading outward, with an oily rainbow sheen reflecting light. The puddle has organic flowing edges. Clean illustration style with bold black outlines, bright yellow and golden colors. Simple clean background. Product illustration style, playful and slightly cartoonish but professional. View from slight angle showing 3D depth. The margarine stick should look soft and glossy, with visible melting occurring at the edges and bottom."

IO.puts("🎨 Generating Margarine logo with FLUX Schnell...")
IO.puts("Prompt: #{prompt}")

case Margarine.generate(prompt, model: :flux_schnell, steps: 4, seed: 42, size: {1024, 1024}) do
  {:ok, image} ->
    IO.puts("✅ Logo generated!")
    case Margarine.Image.save(image, "margarine_logo.png") do
      :ok ->
        IO.puts("✅ Saved to: margarine_logo.png")
        IO.puts("Image shape: #{inspect(Nx.shape(image))}")
      {:error, reason} ->
        IO.puts("❌ Failed to save: #{inspect(reason)}")
    end
  {:error, reason} ->
    IO.puts("❌ Generation failed: #{inspect(reason)}")
end


val_prompts = {}
val_prompts['style'] = [
        'a beautiful mountainous landscape, boulders, fresh water stream, setting sun',
        'the stunning skyline of New York City',
        'fruit hanging from a tree, highly detailed texture, soil, rain, drops, photo realistic, surrealism, highly detailed, 8k macrophotography',
        'the Taj Mahal, stunning wallpaper',
        'A majestic tree rooted in circuits, leaves shimmering with data streams, stands as a beacon where the digital dawn caresses the fog-laden, binary soil—a symphony of pixels and chlorophyll.',
        'A beautiful octopus, with swirling tendrils and a pulsating heart of fiery opal hues, hovers ethereally against a starry void, sculpted through a meticulous flame-working technique.',
        'a stunning image of an aston martin sportscar',
        'the streets of new york city, traffic lights, skyline, setting sun',
        'An ethereal, levitating monolith backlit by a supernova sky, casting iridescent light on the ice-spiked Martian terrain. Neo-futurism, Dali surrealism, wide-angle lens, chiaroscuro lighting.',
        'a portrait of a beautiful young woman',
        'a luminous white lotus blossom floats on rippling waters, green petals',
        'the all seeing eye made of golden feathers, surrounded by waterfall, photorealistic, ethereal aesthetics, powerful',
        'A luminescent glass butterfly, wings shimmering elegantly, depicts deftly the fragility yet adamantine spirit of nature. It encapsulates Atari honkaku technique, glowing embers inside capturing the sun as it gracefully clenches lifes sweet unpredictability',
        'Glass Roots: A luminescent glass sculpture of a fully bloomed rose emerges from a broken marble pedestal, natures resilience triumphant amidst the decay. Shadows cast by a dim overhead spotlight. Delicate veins intertwine the transparent petals, illuminating from within, symbolizing fragilitys steely core.',
        'Eternal Arbor Description: A colossal, life-size tapestry hangs majestically in a dimly lit chamber. Its profound serenity contrasts the grand spectacle it unfolds. Hundreds of intricately woven stitches meticulously portray a towering, ancient oak tree, its knotted branches embracing the heavens. ',
        'In the heart of an ancient forest, a massive projection illuminates the darkness. A lone figure, a majestic mythical creature made of shimmering gold, materializes, casting a radiant glow amidst the towering trees. intricate geometric surfaces encasing an expanse of flora and fauna,',
        'The Silent Of Silicon, a digital deer rendered in hyper-realistic 3D, eyes glowing in binary code, comfortably resting amidst rich motherboard-green foliage, accented under crisply fluorescent, simulated LED dawn.',
        'owl made up of geometric shapes, contours of glowing plasma, black background, dramatic, full picture, ultra high res, octane',
        'A twisting creature of reflective dragonglass swirling above a scorched field amidst a large clearing in a dark forest',
        'what do i say to make me exist, oriental mythical beasts, in the golden danish age, in the history of television in the style of light violet and light red, serge najjar, playful and whimsical, associated press photo, afrofuturism-inspired, alasdair mclellan, electronic media',
        'A towering, rusted iron monolith emerges from a desolate cityscape, piercing the horizon with audacious defiance. Amidst contrasting patches of verdant, natures forgotten touch yearns for connection, provoking intense introspection and tumultuous emotions. vibrant splatters of chaotic paint epitom',
        'A humanoid figure with a luminous, translucent body floats in a vast, ethereal digital landscape. Strands of brilliant, iridescent code rain down, intertwining with the figure. a blend of human features and intricate circuitry, hinting at the merging of organic and digital existence',
        'In the heart of a dense digital forest, a majestic, crystalline unicorn rises. Its translucent, pixelated mane seamlessly transitions into the vibrant greens and golds of the surrounding floating circuit board leaves. Soft moonlight filters through the gaps, creating a breathtaking',
        'Silver mushroom with gem spots emerging from water',
        'Binary Love: A heart-shaped composition made up of glowing binary code, symbolizing the merging of human emotion and technology, incredible digital art, cyberpunk, neon colors, glitch effects, 3D octane render, HD',
        'A labyrinthine maze representing the search for answers and understanding, Abstract expressionism, muted color palette, heavy brushstrokes, textured surfaces, somber atmosphere, symbolic elements',
        'A solitary tree standing tall amidst a sea of buildings, Urban nature photography, vibrant colors, juxtaposition of natural elements with urban landscapes, play of light and shadow, storytelling through compositions',
        ]

val_prompts["face"] = [
            '<concept> as pixel art, 8-bit video game style',
            'painting of <concept> by Vincent van Gogh',
            '<concept> as a statue made of marble',
            '<concept> as a character in a noir graphic novel, under a rain-soaked streetlamp',
            '<concept> portrayed in a famous renaissance painting, replacing Mona Lisas face',
            'a photo of <concept> attending the Oscars, walking down the red carpet with sunglasses',
            #'<concept> as a pop vinyl figure, complete with oversized head and small body',
            '<concept> as a retro holographic sticker, shimmering in bright colors',
            '<concept> as a bobblehead on a car dashboard, nodding incessantly',
            "<concept> captured in a snow globe, complete with intricate details",
            "<concept> as an action figure superhero, lego toy, toy story",
            'a photo of a massive statue of <concept> in the middle of the city',
            'a masterful oil painting portraying <concept> with vibrant colors, brushstrokes and textures',
            'a vibrant low-poly artwork of <concept>, rendered in SVG, vector graphics',
            'an old, vintage, polaroid photograph of <concept>, artsy look',
            '<concept> immortalized as an exquisite marble statue with masterful chiseling, swirling marble patterns and textures',
            ]

val_prompts["object"] = [
            "an intricate wood carving of <concept> in a historic temple",
            "<concept> captured in a snow globe, complete with intricate details",
            "<concept> as a retro holographic sticker, shimmering in bright colors",
            "a painting of <concept> by Vincent van Gogh, impressionism, oil painting, vibrant colors, texture",
            "<concept> as an action figure superhero, lego toy, toy story",
            'a photo of a massive statue of <concept> in the middle of the city',
            'a masterful oil painting portraying <concept> with vibrant colors, thick brushstrokes, abstract, surrealism',
            'an intricate origami paper sculpture of <concept>',
            'a vibrant low-poly artwork of <concept>, rendered in SVG, vector graphics',
            'an artistic polaroid photograph of <concept>, vintage',
            '<concept> immortalized as an exquisite marble statue with masterful chiseling, swirling marble patterns and textures',
            'a colorful and dynamic <concept> mural sprawling across the side of a building in a city pulsing with life',
            "<concept> transformed into a stained glass window, casting vibrant colors in the light",
            "A whimsical papier-mâché sculpture of <concept>, bursting with color and whimsy.",
            "<concept> as a futuristic neon sign, glowing vividly in the night, cyberpunk, vaporwave",
            "A detailed graphite pencil sketch of <concept>, showcasing shadows and depth, pencil drawing, grayscale",
            "<concept> reimagined as a detailed mechanical model, complete with moving parts, metal, gears, steampunk",
            "A vibrant pixel art representation of <concept>, classic 8-bit video game",
            "A breathtaking ice sculpture of <concept>, carved with precision and clarity, ice carving, frozen",
    ]
function Image(el)
  if FORMAT:match 'latex' or FORMAT:match 'beamer' then
    el.src = el.src:gsub('%.svg$', '.pdf')
  end
  return el
end

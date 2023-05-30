## Pretty movies from MUSE datacubes

![A2052](movies/a2052_contsub.gif)

I'll write some proper documentation ASAP, I promise. Meanwhile, you can run `musemovie.py` with: 

```python
python musemovie.py muse_datacube.fits -z 0.004283 -r 6563 -n "M87" -t 45 -s 3.0 -f 25
```


For SITELLE data, you can use the following:

```python
python sitellemovie.py sitelle_datacube.fits -z 0.004283 -r 15245 -n M87
```
using CuArrays, LinearAlgebra, Images, FileIO

MINFLOAT32 = nextfloat( Float32( -Inf ) );

l22d( x1, y1, x2, y2 ) = ( x1 - x2 )^2 + ( y1 - y2 )^2;

cost2dToMatrix( cost, n, m ) = [ cost( i÷n + 1, i%n + 1, j÷n + 1, j%n + 1) for i in 0:(n*m - 1), j in 0:(n*m - 1) ];

zrs( n... ) = CuArrays.fill( 0.0f0, n... )

function entropy( dist )
    dist /= sum( dist ) # we want it to sum to one
    return -(dist' * max.( log.( dist ), MINFLOAT32 ))
end

function wassBarycenters( dists, weights, precision, costMatrix, n_iter, sharpen=false )
	n, k = size( dists )
	nbarys = size( weights, 2 )
	# only accept probability distributions
	totalMass = sum( dists )
	@assert isapprox( totalMass, k ) && totalMass == sum( abs.( dists ) )

	logmus = max.( log.( dists ), MINFLOAT32 )
	# our barycenter should not have higher entropy than the inputs if we sharpen
	entropicBound = -minimum( diag( dists' * logmus ) )

	# renormalizing to avoid underflows in division
	costMatrix = costMatrix ./ max( costMatrix... )
	K = CuArray( exp.( -costMatrix/precision ) )
	weights = CuArray( weights )

	vs, ws, ds = zrs( n, k, nbarys ), zrs( n, k, nbarys ), zrs( n, k, nbarys )
	logbarycenters = zrs( n, nbarys )


	for j in 1:n_iter
		for i in 1:nbarys # CUDA.jl is smart and does this in parallel
			# ws[i] = dists[i] ./ K*vs[i]
			ws[:,:,i] = logmus - log.( K*exp.( vs[:,:,i] ) )
			# ds[i] = vs[i]  .* K*ws[i]
			ds[:,:,i] = vs[:,:,i] + log.( K*exp.( ws[:,:,i] ) )

			logbarycenters[:,i] = ds[:,:,i] * weights[:,i]

			if sharpen && entropy( exp.( logbarycenters[:,i] ) ) > entropicBound
				logbarycenters[:,i] *= 2
			end

			vs[:,:,i] = vs[:,:,i] - ds[:,:,i] .+ logbarycenters[:,i]
		end
	end
	return Array( exp.( logbarycenters ) )
end

function wassBarycenters2d( dists, weights, precision, cost, n_iter, sharpen=false )
	n, m = size( dists[1] )
	nbarys = size( weights, 2)
	C = cost2dToMatrix( cost, n, m )
	dists = CuArray( hcat( ([dist...] for dist in dists)... ) )
	bar = wassBarycenters( dists, weights, precision, C, n_iter, sharpen )
	return [ reshape( bar[:,i], (n, m) ) for i in 1:nbarys ]
end

function imageBarycenter( images, t, sharpen=false )
	inverted = [ 1 .- channelview( img ) for img in images ]
	dists    = [ img/sum( img ) for img in inverted ]
	bar = wassBarycenters2d( dists, [1-t, t], 0.0005, l22d, 100, sharpen )
	return Gray.(1 .- bar[1]./maximum( bar[1] ))
end

function imageBarycenters( images, step=0.05, sharpen=false )
	weights = [ j==1 ? 1 - i : i for j in 1:2, i in 0.0:step:1.0 ]
	inverted = [ 1 .- channelview( Gray.( img ) ) for img in images ]
	dists    = [ img/sum( img ) for img in inverted ]
	barys = wassBarycenters2d( dists, weights, 0.0005, l22d, 100, sharpen )
	imgs = [ Gray.(1 .- bary./maximum( bary )) for bary in barys ]
	return imgs
end

function saveImages( folderName, images )
	mkdir( folderName )
	for (i, img) in enumerate( images )
		save( folderName*"/$i.png", img )
	end
end

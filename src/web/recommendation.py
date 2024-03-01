from audioop import avg
import numpy as np 
import pandas as pd
from web.models import Movie, Myrating
import scipy.optimize 
from django.contrib.auth.models import User


def Myrecommend():
	def normalizeRatings(myY, myR):
    	# The mean is only counting movies that were rated
		Ymean = np.sum(myY,axis=1)/np.sum(myR,axis=1)
		Ymean = Ymean.reshape((Ymean.shape[0],1))
		return myY-Ymean, Ymean
	
	def flattenParams(myX, myTheta):
		return np.concatenate((myX.flatten(),myTheta.flatten()))
    
	def reshapeParams(flattened_XandTheta, mynm, mynu, mynf):
		assert flattened_XandTheta.shape[0] == int(mynm*mynf+mynu*mynf)
		reX = flattened_XandTheta[:int(mynm*mynf)].reshape((mynm,mynf))
		reTheta = flattened_XandTheta[int(mynm*mynf):].reshape((mynu,mynf))
		return reX, reTheta

	def cofiCostFunc(myparams, myY, myR, mynu, mynm, mynf, mylambda = 0.):
		myX, myTheta = reshapeParams(myparams, mynm, mynu, mynf)
		term1 = myX.dot(myTheta.T)
		term1 = np.multiply(term1,myR)
		cost = 0.5 * np.sum( np.square(term1-myY) )
    	# for regularization
		cost += (mylambda/2.) * np.sum(np.square(myTheta))
		cost += (mylambda/2.) * np.sum(np.square(myX))
		return cost

	def cofiGrad(myparams, myY, myR, mynu, mynm, mynf, mylambda = 0.):
		myX, myTheta = reshapeParams(myparams, mynm, mynu, mynf)
		term1 = myX.dot(myTheta.T)
		term1 = np.multiply(term1,myR)
		term1 -= myY
		Xgrad = term1.dot(myTheta)
		Thetagrad = term1.T.dot(myX)
    	# Adding Regularization
		Xgrad += mylambda * myX
		Thetagrad += mylambda * myTheta
		return flattenParams(Xgrad, Thetagrad)

	def getUserSimilarityScores(prediction_matrix, average_rating):
	

		# Calculating the similarity scores
		sim_scores = np.empty((len(prediction_matrix), len(prediction_matrix)))
		for i in range(len(prediction_matrix)):
			for j in range(i + 1, len(prediction_matrix)):
				numerator = np.sum((prediction_matrix[i,:]-average_rating)*(prediction_matrix[j,:]-average_rating))
				denominator = np.sqrt(np.sum(prediction_matrix[i,:]**2)) * np.sqrt(np.sum(prediction_matrix[j,:]**2))
				
				if denominator == 0 :
					sim_scores[i,j] = 0
				else:
					sim_scores[i, j] = numerator / denominator
				sim_scores[j, i] = sim_scores[i, j]
				print("UserIds[",i,",",j,"]",sim_scores[i,j])

		print("i:",len(prediction_matrix),"j:", len(prediction_matrix[0]))
		
		return sim_scores

	myRatingTable = Myrating.objects.all().values()
	movieTable = Movie.objects.all().values()
	userTable = User.objects.all().values()
	mynf = 10
	# Extract unique movie and user IDs
	unique_movie_ids = [movie['id'] for movie in movieTable]
	unique_user_ids = [user['id'] for user in userTable]
	all_ratings = [rating['rating'] for rating in myRatingTable]
	average_rating = sum(all_ratings) / len(all_ratings) if len(all_ratings) > 0 else 0

	print("average::::",average_rating)
	Y = np.zeros((len(unique_user_ids), len(unique_movie_ids)))  ## makes a 0 matrix


	for i, movie_id in enumerate(unique_movie_ids):
		for j, user_id in enumerate(unique_user_ids):
			ratings = [rating['rating'] for rating in myRatingTable if rating['user_id'] == user_id and rating['movie_id'] == movie_id]
			
			# Assuming a user can provide only one rating for a specific movie
			if ratings:
				Y[j, i] = ratings[0]


	R = np.zeros((len(unique_user_ids), len(unique_movie_ids)))  ## makes a 0 matrix
	for i in range(Y.shape[0]):
		for j in range(Y.shape[1]):
			if Y[i][j]!=0:
				R[i][j]=1

	Ynorm, Ymean = normalizeRatings(Y,R)
	X = np.random.rand(len(unique_user_ids),mynf)
	Theta = np.random.rand(len(unique_movie_ids),mynf)
	myflat = flattenParams(X, Theta)
	mylambda = 12.2
	result = scipy.optimize.fmin_cg(cofiCostFunc,x0=myflat,fprime=cofiGrad,args=(Y,R,len(unique_movie_ids),len(unique_user_ids),mynf,mylambda),maxiter=40,disp=True,full_output=True)
	resX, resTheta = reshapeParams(result[0], len(unique_user_ids), len(unique_movie_ids), mynf)
	prediction_matrix = resX.dot(resTheta.T)
	print(prediction_matrix)
	getUserSimilarityScores(prediction_matrix,average_rating)
	return prediction_matrix,Ymean
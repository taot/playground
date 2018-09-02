package trie;

import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

public class WordSquaresTest {

    WordSquares obj = new WordSquares();

    @Test
    public void test_example_1() {
        String[] input = {"area","lead","wall","lady","ball"};
        List<List<String>> output = obj.wordSquares(input);
        List<List<String>> expected = Arrays.asList(
                Arrays.asList("ball", "area", "lead", "lady"),
                Arrays.asList("wall", "area", "lead", "lady")
        );
        Assert.assertEquals(expected, output);
    }

    @Test
    public void test_example_2() {
        String[] input = {"abat","baba","atan","atal"};
        List<List<String>> output = obj.wordSquares(input);
        List<List<String>> expected = Arrays.asList(
                Arrays.asList("baba", "abat", "baba", "atal"),
                Arrays.asList("baba", "abat", "baba", "atan")
        );
        Assert.assertEquals(expected, output);
    }

    @Test
    public void test_timeout_1() {
        String[] input = {"buff","ulus","buns","rump","cuts","stum","murk","wuss","putt","pubs","bust","chub","burp","bubs","suns","puns","buhr","ughs","mums","cunt","bhut","guff","pung","phut","flux","snub","ruts","vugg","turd","hung","tups","xyst","puny","curr","curf","typy","busk","byrl","cusp","pups","pulp","duns","dunk","tugs","dull","bury","murr","slum","mumm","jugs","burn","purl","curl","runt","spry","typp","fugu","dunt","mump","cuds","juju","sudd","nuts","culm","dumb","gyps","buzz","surf","putz","tung","tuns","puds","urns","tuck","duct","hugs","jump","bums","lulu","myth","rynd","undy","hunh","gulf","guts","lutz","burl","lump","dung","gull","gush","bunk","tusk","dups","stub","gust","curs","juts","swum","luff","subs","psst","syph","junk","funs","flub","hurt","burg","muck","buts","furl","such","mull","huff","chug","kuru","dubs","guls","drum","bunt","blub","rhus","buss","hump","rust","stud","fund","cubs","plum","punk","brut","cuff","sugh","wyns","pugh","cuss","buhl","hulk","burd","lurk","hymn","shun","yurt","puts","scum","luny","muns","lung","glug","hunk","guru","cyst","sump","slut","bull","gnus","thud","spur","cups","hunt","busy","yups","durr","turf","guck","full","sulk","purr","smut","curn","butt","suqs","duly","fuds","curb","chum","husk","upby","crud","grum","mugg","scut","thru","vugs","urbs","pump","bulb","smug","kudu","sync","punt","gyms","ruly","frug","crus","tuft","must","bund","ruff","rugs","push","mush","drys","rung","slub","bulk","dust","puck","shut","yuch","gunk","shul","tuts","purs","luck","whys","surd","rubs","dump","numb","thus","buck","duff","duds","gums","muds","sums","hull","thug","bung","musk","lynx","dusk","huts","puff","bump","wych","fugs","runs","jury","tump","bumf","huns","lunk","guys","sunk","lull","buds","glut","bush","knur","pull","sung","muff","funk","plug","ugly","snug","bugs","rush","lush","wynd","furs","curd","suds","hurl","muss","umps","slug","rums","urds","bunn","slur","blur","fuzz","stun","luvs","pugs","yuck","suss","scry","turk","mutt","muts","glum","sulu","mugs","grub","much","plus","buys","tuff","burs","cull","just","urus","drub","hums","null","fury","tsks","scup","lunt","sunn","tutu","lugs","curt","vugh","sups","nuns","trug","durn","nurd","puss","lums","mumu","nurl","cusk","drug","cult","gulp","club","puls","brrr","scud","huck","fuss","turn","fuck","tubs","ruth","ruby","duty","nubs","cwms","guvs","crux","suck","spud","rusk","dugs","wynn","hubs","futz","tush","rudd","pfft","hush","burr","hyps","ruck","fumy","yuks","duck","lust","guns","spun","fubs","flus"};
        List<List<String>> output = obj.wordSquares(input);
        System.out.println(output);
        Assert.assertEquals(450, output.size());
    }
}

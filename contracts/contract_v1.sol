pragma solidity >=0.7.0 <0.9.0;

import "@openzeppelin/contracts/token/ERC721/extensions/ERC721Enumerable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@1001-digital/erc721-extensions/contracts/RandomlyAssigned.sol";

contract NFT is ERC721, Ownable, RandomlyAssigned {
    using Strings for uint256;

    string public baseURI;

    string public baseExtension = ".json";

    uint256 public cost = 0.05 ether; // base price for 1 token

    uint256 public maxSupply = 10000; // total number of tokens

    uint256 public maxMintAmount = 20; // max amount of tokens for batch mint, 20 is probably too high

    bool public paused = false;

    mapping(address => bool) public whitelisted;

    constructor(
        string memory _name,
        string memory _symbol,
        string memory _initBaseURI
    ) ERC721(_name, _symbol) 
    RandomlyAssigned(10000,1)
    {
        setBaseURI(_initBaseURI);
        mint(msg.sender, 20); // mint 20 tokens when publishing the contract, remove otherwise
    }

    // internal
    function _baseURI() internal view virtual override returns (string memory) {
        return baseURI;
    }

    // public
    function mint(address _to, uint256 _mintAmount) public payable {
        require(!paused, "Sale must be active");
        require(_mintAmount > 0, "At least 1 token needs to be minted");
        require(_mintAmount <= maxMintAmount, "Cannot mint more than 20 token at a time");
        require(tokenCount() + _mintAmount <= totalSupply(), "Purchase would exceed max amount of tokens");
        require(availableTokenCount() - _mintAmount >= 0, "Cannot mint more than the available token count");

        if (msg.sender != owner()) {
            if(whitelisted[msg.sender] != true) {
            require(msg.value >= cost * _mintAmount, "Ether value is not correct");
            }
        }

        for (uint256 i = 1; i <= _mintAmount; i++) {
            uint256 id = nextToken();
            _safeMint(_to, id);
        }
    }

    // return metadata location for speciifed Id
    function tokenURI(uint256 tokenId)
        public
        view
        virtual
        override
        returns (string memory)
    {
        require(
        _exists(tokenId),
        "ERC721Metadata: URI query for nonexistent token"
        );

        string memory currentBaseURI = _baseURI();
        return bytes(currentBaseURI).length > 0
            ? string(abi.encodePacked(currentBaseURI, tokenId.toString(), baseExtension))
            : "";
    }

    //only owner, setter for token base price
    function setCost(uint256 _newCost) public onlyOwner {
        cost = _newCost;
    }

    //only owner, setter for max amount for batch mint
    function setmaxMintAmount(uint256 _newmaxMintAmount) public onlyOwner {
        maxMintAmount = _newmaxMintAmount;
    }

    //only owner, setter for metadata base URI
    function setBaseURI(string memory _newBaseURI) public onlyOwner {
        baseURI = _newBaseURI;
    }

    //only owner, setter for metadata base extension
    function setBaseExtension(string memory _newBaseExtension) public onlyOwner {
        baseExtension = _newBaseExtension;
    }

    function pause(bool _state) public onlyOwner {
        paused = _state;
    }

    function whitelistUser(address _user) public onlyOwner {
        whitelisted[_user] = true;
    }

    function removeWhitelistUser(address _user) public onlyOwner {
        whitelisted[_user] = false;
    }

    // Used to withdraw funds from the contract and send them to owner wallet
    function withdraw() public payable onlyOwner {
        require(payable(msg.sender).send(address(this).balance));
    }
}